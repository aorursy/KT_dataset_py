# Import the necessary packages
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from xgboost.sklearn import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
# Import and read dataset
input_ = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(input_)

df.head(10)
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.show()
df.describe()
x = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']

model = XGBClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
for i in range(0,len(df.columns)):
    print("{} = {}".format(i,df.columns[i]))
# Delete outlier
df = df[df['ejection_fraction']<70]
inp_data = df.drop(df[['DEATH_EVENT']], axis=1)
#inp_data = df.iloc[:,[11,7,4,0,1,8]]
out_data = df[['DEATH_EVENT']]

X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=0, shuffle=True)

## Applying Transformer
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
## X_train, X_test, y_train, y_test Shape

print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)
# I coded this method for convenience and to avoid writing the same code over and over again

def result(clf):
    clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(X_test, y_test)], verbose=False)
    y_pred = clf.predict(X_test)
    
    print('Accuracy Score    : {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('XGBoost f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
    print('XGBoost precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
    print('XGBoost recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
    print("XGBoost roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
    print("\n",classification_report(y_pred, y_test))
    
    plt.figure(figsize=(6,6))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
    plt.title("XGBoost Confusion Matrix (Rate)")
    plt.show()
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
    plt.title("XGBoost Confusion Matrix (Number)")
    plt.show()
    
    
def report(**params):
    scores = [] 
    for i in range(0,250): # 250 samples
        X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, shuffle=True)
        sc = StandardScaler()
        clf = XGBClassifier(**params)
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(X_test, y_test)], verbose=False)
        scores.append(accuracy_score(clf.predict(X_test), y_test)) 
        
    Importance = pd.DataFrame({'Importance':clf.feature_importances_*100},index=df.columns[:12])
    Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='lightblue')
    plt.xlabel('Importance for variable');
    plt.hist(scores)
    plt.show()
    print("Best Score: {}\nMean Score: {}".format(np.max(scores), np.mean(scores)))
report()
param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

clf = XGBClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 5,
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 5,
)
param_grid = {
    'gamma': [i/10.0 for i in range(0,8)]
}

clf = XGBClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
)
param_grid = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree': [i/10.0 for i in range(7,15)]
}

clf = XGBClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
    seed=0
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
)
param_grid = {
 'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}

clf = XGBClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
    reg_alpha= 0.01,
    seed=0
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
    reg_alpha= 0.01
)
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
    reg_alpha= 0.01,
    learning_rate=0.1,
    objective= 'binary:logistic',
    seed= 0
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 5,
    gamma = 0.5,
    colsample_bytree= 0.9,
    subsample= 0.8,
    reg_alpha= 0.01,
    learning_rate=0.1,
    objective= 'binary:logistic',
)

param_grid = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma': [i/10.0 for i in range(0,8)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree': [i/10.0 for i in range(7,15)],
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'objective':['binary:logistic']
}

clf = XGBClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = XGBClassifier(
    max_depth= 5,
    min_child_weight= 3,
    gamma = 0.2,
    colsample_bytree= 0.7,
    subsample= 0.6,
    reg_alpha= 0.01,
    learning_rate=0.1,
    objective= 'binary:logistic',
    seed= 0
)

result(clf)
report(
    max_depth= 5,
    min_child_weight= 3,
    gamma = 0.2,
    colsample_bytree= 0.7,
    subsample= 0.6,
    reg_alpha= 0.01,
    learning_rate=0.01,
    objective= 'binary:logistic',
)