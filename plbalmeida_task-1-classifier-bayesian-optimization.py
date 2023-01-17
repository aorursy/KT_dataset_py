# libraries, classes etc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='ticks', color_codes=True)
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
# loading data
df = pd.read_excel('../input/covid19/dataset.xlsx')

df = df.set_index('Patient ID') # patients as index

df = df.drop([
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)'
], axis=1)

# first observations
df.head()
# dimension
df.shape
# numeric conversion
df['Urine - pH'] = df['Urine - pH'].replace('Não Realizado', 0).astype('float')
df['Urine - Leukocytes'] = df['Urine - Leukocytes'].replace('<1000', 0).astype('float')

def func(x):
    if x == 'negative':
        return 0
    elif x == 'positive':
        return 1

df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].apply(func) 
# object features
df['Patient age quantile'] = df['Patient age quantile'].astype('str')

object_columns = df.select_dtypes(include='object').columns

# keep numeric features (aka hemogran)
df_num = df.drop(object_columns, axis=1)
df_num = df_num.dropna(thresh=len(df_num.columns) * 0.3) # keep rows with 30% of NaN

# dimension
df_num.shape
print('Negative COVID-19 cases: ', len(df_num[df_num['SARS-Cov-2 exam result'] == 0]))
print('Positive COVID-19 cases: ', len(df_num[df_num['SARS-Cov-2 exam result'] == 1]))
# new dataframe
df = df_num.copy()

# features with at least 1 observation filled 
df = df.dropna(axis=1, how='all')

# dimension
df.shape
# predictors
X = df.drop(['SARS-Cov-2 exam result'], axis=1)

# fill missing values for RandomUnderSampler
imputer = KNNImputer(n_neighbors=3)
imputer.fit(X)
X = pd.DataFrame(imputer.transform(X), index=X.index, columns=X.columns)

# target
y = df['SARS-Cov-2 exam result']

# standardize features
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
# feature selection with ExtraTreesClassifier  
model = ExtraTreesClassifier(n_estimators=100)
model.fit(X, y)

# feature selection and preparation for feature importance plot 
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['Feature importance'])

treshold = feature_importances['Feature importance'].quantile(q=0.5) # treshold (median)

feature_importances = feature_importances[feature_importances['Feature importance'] >= treshold] # cut above treshold
features = feature_importances.index
feature_importances = feature_importances.sort_values('Feature importance', ascending=False)

plt.subplots(figsize=(10,10))
sns.barplot(x=feature_importances['Feature importance'], y=feature_importances.index, data=feature_importances, color='b')
plt.title('Feature Importance', size=15)
plt.xlabel('%')
plt.grid(axis='x')
plt.show()
# dataframe with best predictors
X = X[list(features)]

# dimension
X.shape
# train/test split
msk = np.random.rand(len(df)) < 0.6
X_train, y_train = X[msk], y[msk]
X_test, y_test = X[~msk], y[~msk]

# RandomUnderSampler
sampler = RandomUnderSampler(random_state=123)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
# classifier name
names = [
    'Random forest',
    'XGBoost',
    'Multi-layer Perceptron'
]

# classifier class
classifiers = [
    RandomForestClassifier(n_jobs=-1, random_state=123),
    XGBClassifier(n_jobs=-1, random_state=123),
    MLPClassifier()
]

# hyperparameter space
parameters = [
           
    # Random forest
    {
        'bootstrap' : Integer(0, 1),
        'max_depth' : Integer(5, 15),
        'max_features' : Categorical(['auto', 'sqrt']),
        'min_samples_leaf' : Integer(1, 4),
        'min_samples_split' : Integer(2, 10),
        'n_estimators': Integer(50, 100)
    },
    
    # XGBoost
    {
        'learning_rate' : Real(0.05, 0.31, prior='log-uniform'),
        'max_depth' : Integer(5, 15),
        'min_child_weight' : Integer(1, 8),
        'colsample_bytree' : Real(0.3, 0.8, prior='log-uniform'),
        'subsample' : Real(0.8, 1, prior='log-uniform'),
        'n_estimators' : Integer(50, 100)
    },
    
    # Multi-layer Perceptron
    { 
        'activation' : Categorical(['identity', 'logistic', 'tanh', 'relu']),
        'solver' : Categorical(['lbfgs', 'sgd', 'adam']),
        'alpha' : Real(1e-6, 1e-2, prior='log-uniform'),
        'learning_rate' : Categorical(['constant', 'invscaling', 'adaptive']),
        'max_iter' : Integer(100, 500)
    }
]
# classifier + bayesian optimization of hyperparameters
for name, clf, param in zip(names, classifiers, parameters):
     
    opt = BayesSearchCV(clf, param, scoring='precision', n_iter=50, cv=10, n_jobs=-1, refit=True, random_state=123)
    opt.fit(X_train_resampled, y_train_resampled)
    
    cv_results = opt.cv_results_
    best_params = opt.best_params_
    
    y_pred = opt.best_estimator_.predict(X_test)
    
    print(name)
    
    print('')
    
    plt.boxplot(cv_results['mean_test_score'])
    plt.title('Precision - Train')
    plt.show()
    
    print('')
    
    print('AUC score: ', roc_auc_score(y_test, y_pred))
    
    print('')
    
    print(classification_report(y_test, y_pred))
    
    print('')
    print('-----------------------------------------------------------------------')
    print('')