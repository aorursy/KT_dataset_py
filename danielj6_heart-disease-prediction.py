import numpy as np

np.random.seed(35702)

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
# Load data

data = pd.read_csv(r'../input/heart-disease-cleveland-uci/heart_cleveland_upload.csv')
# First, we'll create a copy of the data for exploratory analysis and rename the values of the categorical features and the target to their meanings

data_explore = data.copy(deep=True)

data_explore['sex'] = data_explore['sex'].map({0: 'female', 1: 'male'})

data_explore['cp'] = data_explore['cp'].map({0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'})

data_explore['fbs'] = data_explore['fbs'].map({0: '<= 120 mg/dl', 1: '> 120 mg/dl'})

data_explore['restecg'] = data_explore['restecg'].map({0: 'normal', 1: 'ST-T abnormality', 2: 'Left Ventricular Hypertrophy'})

data_explore['exang'] = data_explore['exang'].map({0: 'No exercise-induced angina', 1: 'Exercise-induced angina'})

data_explore['slope'] = data_explore['slope'].map({0: 'upsloping', 1: 'flat', 2: 'downsloping'})

data_explore['thal'] = data_explore['thal'].map({0: 'normal', 1: 'fixed defect', 2: 'reversable defect'})

data_explore['condition'] = data_explore['condition'].map({0: 'no heart disease', 1: 'heart disease'})
data_explore.head(25)
print(data_explore.shape)
def summarize_data(data: pd.DataFrame) -> pd.DataFrame:

    """Return DataFrame summarizing data for each feature"""

    summary = data.describe(include='all').transpose()

    summary['dtype'] = data.dtypes

    summary['missing'] = data.isnull().sum()

    summary['zeros'] = (data == 0).astype(int).sum()

    summary['skewness'] = data.skew()

    return summary
summarize_data(data_explore)
sns.countplot(data_explore['condition'])
numeric_features = ['age', 'trestbps', 'chol', 'thalach']

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

target = 'condition'
sns.pairplot(data_explore[numeric_features+['condition']], hue='condition')

plt.suptitle('')
plt.figure(figsize=(8, 8))

sns.heatmap(data_explore.corr(), annot=True, fmt='.1f', cmap='coolwarm', center=0)
for feature in numeric_features:

    plt.figure()

    sns.boxplot(data_explore[feature])
for feature in numeric_features:

    plt.figure()

    sns.regplot(x=feature, y=target, data=data, x_bins=5)
f = plt.figure()

data.corr(method='pearson')['condition'].sort_values(ascending=True).drop(['condition'], axis=0).plot.barh()

plt.xlabel('Pearson Correlation with Presence of Heart Disease')
for feature in categorical_features:

    f, axs = plt.subplots(1, 2)

    sns.countplot(data_explore[feature], ax=axs[0])

    sns.pointplot(x=feature, y=data['condition'], data=data_explore)

    for ax in axs:

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
from typing import Dict



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate



from sklearn.base import BaseEstimator

from sklearn.pipeline import Pipeline



from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier



import lightgbm as lgbm
X, y = data.drop(['condition'], axis=1), data['condition']
# split off a small amount of test data for getting the expected

# performance of the chosen model in the next section.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
def evaluate_model(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 10) -> Dict[str, float]:

    """Print and return cross validation of model"""

    scoring = 'accuracy'

    scores = cross_validate(estimator, X, y, return_train_score=True, cv=cv, scoring=scoring)

    train_mean, train_std = scores['train_score'].mean(), scores['train_score'].std()

    print(f'Train accuracy: {train_mean} ({train_std})')

    val_mean, val_std = scores['test_score'].mean(), scores['test_score'].std()

    print(f'Validation accuracy: {val_mean} ({val_std})')

    fit_mean, fit_std = scores['fit_time'].mean(), scores['fit_time'].std()

    print(f'Fit time: {fit_mean} ({fit_std})')

    score_mean, score_std = scores['score_time'].mean(), scores['score_time'].std()

    print(f'Score time: {score_mean} ({score_std})')

    result = {

        'Train Accuracy': train_mean,

        'Train std': train_std,

        'Validation Accuracy': val_mean,

        'Validation std': val_std,

        'Fit Time (s)': fit_mean,

        'Score Time (s)': score_mean,

    }

    return result
# define encoding of categorical features

encoder = ColumnTransformer([

    ('onehot', OneHotEncoder(handle_unknown='ignore'), np.isin(X_train.columns.tolist(), categorical_features)), 

])
dummy = DummyClassifier(strategy='most_frequent')

dummy_evaluation = evaluate_model(dummy, X_train, y_train)
lr = Pipeline([

    #('encode', encoder),

    ('scale', StandardScaler()),

    ('lr', LogisticRegression())

])

lr_evaluation = evaluate_model(lr, X_train, y_train)
gnb = Pipeline([

    #('encode', encoder),

    ('scale', StandardScaler()),

    ('gnb', GaussianNB()),

])

gnb_evaluation = evaluate_model(gnb, X_train, y_train)
knn = Pipeline([

    ('encode', encoder),

    ('scale', StandardScaler()),

    ('knn', KNeighborsClassifier()),

])

knn_evaluation = evaluate_model(knn, X_train, y_train)
dt = Pipeline([

    ('encode', encoder),

    ('scale', StandardScaler()),

    ('tree', DecisionTreeClassifier(max_depth=6)),

])

dt_evaluation = evaluate_model(dt, X_train, y_train)
et = Pipeline([

    ('encode', encoder),

    ('scale', StandardScaler()),

    ('extra tree', ExtraTreesClassifier(max_depth=4))

])

et_evaluation = evaluate_model(et, X_train, y_train)
lgb = Pipeline([

    ('encode', encoder),

    ('scale', StandardScaler()),

    ('LightGBM', lgbm.LGBMClassifier(max_depth=1)),

])



lgbm.LGBMClassifier()

lgb_evaluation = evaluate_model(lgb, X_train, y_train)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier



X_encoded = encoder.fit_transform(X_train)



def create_nn_model() -> Sequential:

    """Create neural network model"""

    model = Sequential()

    #input_dim = X_encoded.shape[1]

    input_dim = X_train.shape[1]

    model.add(Dense(50, input_dim=input_dim, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(20, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy')

    return model



nn = Pipeline([

    #('encoder', encoder),

    ('scale', StandardScaler()),

    ('nn', KerasClassifier(build_fn=create_nn_model, epochs=300, verbose=0)),

])

nn_evaluation = evaluate_model(nn, X_train, y_train, cv=3)    # use lower cv since fit is slower
# summarize model performances

pd.DataFrame({

    'Dummy': dummy_evaluation,

    'Logistic Regression': lr_evaluation,

    'Gaussian Naive Bayes': gnb_evaluation,

    'K-Nearest Neighbors': knn_evaluation,

    'Decision Tree': dt_evaluation,

    'Extra Trees': et_evaluation,

    'LightGBM': lgb_evaluation,

    'Neural Network': nn_evaluation,

}).transpose().sort_values('Validation Accuracy', ascending=False)
from sklearn.model_selection import GridSearchCV





# wide parameter range

lr = Pipeline([

    ('scale', StandardScaler()),

    ('lr', LogisticRegression()),

])



param_grid = {'lr__C': np.logspace(-3, 3, 20)}

tuned_lr = GridSearchCV(lr, return_train_score=True, param_grid=param_grid, scoring='accuracy', cv=10)

tuned_lr.fit(X_train, y_train)

results = tuned_lr.cv_results_
plt.figure()

plt.semilogx(param_grid['lr__C'], results['mean_train_score'], label='Train')

plt.semilogx(param_grid['lr__C'], results['mean_test_score'], label='Test')

plt.legend()
# narrower parameter range

param_grid = {'lr__C': np.logspace(-2.5, 0, 50)}

tuned_lr = GridSearchCV(lr, return_train_score=True, param_grid=param_grid, scoring='accuracy', cv=10)

tuned_lr.fit(X_train, y_train)

results = tuned_lr.cv_results_
plt.figure()

plt.semilogx(param_grid['lr__C'], results['mean_train_score'], label='Train')

plt.semilogx(param_grid['lr__C'], results['mean_test_score'], label='Test')

plt.legend()
final_model = Pipeline([

    ('scale', StandardScaler()),

    ('lr', LogisticRegression(C=10**(-1.5))),

])

final_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



test_score = accuracy_score(y_test, final_model.predict(X_test))

print(f'Accuracy on Test Set of {test_score}')
print(classification_report(y_test, final_model.predict(X_test)))
from sklearn.metrics import roc_curve

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import plot_precision_recall_curve

from sklearn.metrics import plot_roc_curve



fpr, tpr, _ = roc_curve(y_test, final_model.predict_proba(X_test)[:, 1])
plot_confusion_matrix(final_model, X_test, y_test)
plot_precision_recall_curve(final_model, X_test, y_test)
from sklearn.inspection import plot_partial_dependence

plt.figure(figsize=(12, 16))

plot_partial_dependence(final_model, X_train, X_train.columns, ax=plt.gca())

plt.tight_layout()