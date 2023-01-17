# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc, precision_score, recall_score
import lightgbm as lgb
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data_df.head()
data_df.isnull().sum()
sns.countplot(data_df['Outcome'])
sns.boxplot(data=data_df, y='Glucose', x='Outcome')
np.log(data_df['Pregnancies'])
sns.boxplot(data=data_df, y='Pregnancies', x='Outcome')
sns.boxplot(y=np.log(data_df['Pregnancies'] + 1), x=data_df['Outcome'])
sns.boxplot(data=data_df, y='BloodPressure', x='Outcome')
sns.boxplot(y=np.log(data_df['SkinThickness'] + 1), x=data_df['Outcome'])
sns.boxplot(data=data_df, y='SkinThickness', x='Outcome')
sns.boxplot(data=data_df, y='Insulin', x='Outcome')
sns.boxplot(data=data_df, y='BMI', x='Outcome')
sns.boxplot(data=data_df, y='Age', x='Outcome')
sns.boxplot(y=np.log(data_df['DiabetesPedigreeFunction'] + 1), x=data_df['Outcome'])
sns.boxplot(data=data_df, y='DiabetesPedigreeFunction', x='Outcome')
data_df.groupby('Outcome').describe()
def build_model(model):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return model
def build_lgbm(X_train, X_test, y_train, y_test):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'colsample_bytree': 0.7, 
        'max_depth': 15, 
        'min_split_gain': 0.3, 
        'n_estimators': 400, 
        'num_leaves': 50, 
        'reg_alpha': 1.1, 
        'reg_lambda': 1.1, 
        'subsample': 0.8, 
        'subsample_freq': 20
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test)
    return lgb_train, lgb_eval, params
def perform_SMOTE(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
def compute_metrics(y_pred, y_test, threshold):
    binary_pred = np.array([1 if pred > threshold else 0 for pred in y_pred])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print('AUC Score: ', auc(fpr, tpr))
    print('AUC ROC Score: ', roc_auc_score(y_test, y_pred))
    print('F1 Score: ', f1_score(y_test, binary_pred))
    print('Accuracy Score: ', accuracy_score(y_test, binary_pred))
    print('Precision Score: ', precision_score(y_test, binary_pred))
    print('Recall Score: ', recall_score(y_test, binary_pred))
    print('tn, fp, fn, tp: ', confusion_matrix(y_test, binary_pred).ravel())
    print(sns.heatmap(confusion_matrix(y_test, binary_pred), annot=True, 
                      xticklabels=['Pred 0', 'Pred 1'], 
                      yticklabels=['Actual 0', 'Actual 1']));
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
        pred = fitted_model.predict_proba(X_test_data)
    else:
        pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred
y = data_df['Outcome']
X = data_df.drop('Outcome', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
X_train, y_train = perform_SMOTE(X_train, y_train)
lgb_train, lgb_eval, params = build_lgbm(X_train, X_test, y_train, y_test)
gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
compute_metrics(y_pred, y_test, 0.5)
model = lgb.LGBMClassifier()
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}

model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 param_grid, cv=5, scoring_fit='f1')

print(model.best_score_)
print(model.best_params_)
compute_metrics(pred, y_test, 0.5)
model = build_model(LogisticRegression(max_iter=1000))
model.fit(X_train,y_train)
compute_metrics(model.predict_proba(X_test)[:,1], y_test, 0.5)
model = build_model(LogisticRegression(max_iter=1000))
param_grid = {
    'C': [0.01, 0.1, 1, 10],
}
model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 param_grid, cv=5, scoring_fit='f1', do_probabilities= True)

print(model.best_params_)

compute_metrics(pred[:,1:], y_test, 0.5)
model = build_model(SVC())
model.fit(X_train,y_train)
compute_metrics(model.predict(X_test), y_test, 0.5)
model = build_model(SVC())
param_grid = {
    'C': [0.01, 0.1, 1, 10],
}
model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 param_grid, cv=5, scoring_fit='f1', do_probabilities= False)

print(model.best_params_)

compute_metrics(pred, y_test, 0.5)
model = build_model(RandomForestClassifier())
model.fit(X_train,y_train)
compute_metrics(model.predict_proba(X_test)[:,1:], y_test, 0.5)
model = build_model(RandomForestClassifier())
param_grid = {
    'n_estimators': [100, 1000, 10000],
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 7]
}
model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                 param_grid, cv=2, scoring_fit='f1', do_probabilities= True)

print(model.best_params_)

compute_metrics(pred[:,1:], y_test, 0.5)