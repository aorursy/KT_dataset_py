# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
df = pd.read_csv("/kaggle/input/datasets-for-churn-telecom/cell2celltrain.csv")
df.head(10)
df.shape
df.dtypes
# df.count()
df['Churn'].value_counts()
for x in df.columns:
    if df[x].isnull().sum() != 0:
        print(x, df[x].isnull().sum())
df.drop(['CustomerID'], axis=1, inplace=True)
df = df.dropna()
# Can handle in a different way
df['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df['Churn'].replace(to_replace='No',  value=0, inplace=True)
# List of categoricals
categoricals = list()
for x in df.columns:
    if df[x].dtype == 'object':
        categoricals.append(x)
df[categoricals].nunique()
def plot_val_counts(df, col=''):
    plt.figure(figsize=(5,5))
    plt.grid(True)
    plt.bar(df[col][df.Churn==1].value_counts().index, 
            df[col][df.Churn==1].value_counts().values)
    plt.title(f'{col}')
    plt.xticks(rotation=-90)
# df['HandsetPrice'] = df['HandsetPrice'].replace("Unknown", np.nan)
# df = df.fillna(method='ffill')
# df['HandsetPrice'] = df['HandsetPrice'].replace("Unknown", 0)
# df['HandsetPrice'] = df['HandsetPrice'].astype(int)
plot_val_counts(df, col='HandsetPrice')
plot_val_counts(df, col='CreditRating')
plot_val_counts(df, col='Occupation')
plot_val_counts(df, col='PrizmCode')
# graph = df[["Occupation", "MonthlyRevenue"]]
# ax = sns.barplot(x="Occupation", y="MonthlyRevenue", data=graph)
numericals = [x for x in df.columns if x not in categoricals]

plt.figure(figsize=(15,8))
df[numericals].corr()['Churn'].sort_values(ascending = False).plot(kind='bar')
correlated_features = set()
correlation_matrix = df[numericals].corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname1 = correlation_matrix.columns[i]
            colname2 = correlation_matrix.columns[j]
#             print (correlation_matrix.columns[i] + ' and ' + correlation_matrix.columns[j])
            if colname1 != 'Churn' and colname2 != 'Churn':
                if abs(correlation_matrix['Churn'][colname1]) > abs(correlation_matrix['Churn'][colname2]):
                    correlated_features.add(colname2)
                else:
                    correlated_features.add(colname1)
print(correlated_features)
df.drop(correlated_features, axis=1, inplace=True)
df_dummies = pd.get_dummies(df, drop_first=True)
X = df_dummies.drop(['Churn'], axis=1)
y = df_dummies['Churn']


# Standardize the features
features = X.columns.values
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features
# X.head()
# X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16, stratify=y)
# y_train.sum()/y_train.shape[0]
from sklearn.linear_model import LassoCV

lasso = LassoCV().fit(X_train, y_train)
importance = np.abs(lasso.coef_)
print(len([x for x in importance if x != 0]))
# feature_names = np.array(X_train.columns.values)
# plt.bar(height=importance, x=feature_names)
# plt.title("Feature importances via coefficients")
# plt.show()
# df['IncomeGroup'] = df['IncomeGroup'].astype(str)
# df.dtypes
# df['CreditRating'] = df['CreditRating'].str[:1]
# df['CreditRating'] = df['CreditRating'].astype(int)
# # df['ServiceArea'] = df['ServiceArea'].str[-3:]
# # df['ServiceArea'] = df['ServiceArea'].astype(int)
# df.drop(['ServiceArea'], axis=1, inplace=True)
# # for col in df.columns:
# #     if df[col].dtype != np.dtype('int64') and \
# #         df[col].dtype != np.dtype('float64'):
# #         df[col] = pd.factorize(df[col])[0]
# df.describe()
# # Downsampling
# df_ds, _ = train_test_split(df[df['Churn'] == 0], test_size=0.6)
# df_ds = pd.concat([df_ds, df[df['Churn'] == 1]])
# df_ds.shape
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(lasso).fit(X_train, y_train)

selected_feat = X_train.columns[(sfm.get_support())]
print(selected_feat)

X_train = sfm.transform(X_train)
X_test = sfm.transform(X_test)
# Running logistic regression model
lr = LogisticRegression()

grid_values = {'C':[0.001,0.01,1,10]}#, 'penalty': ['l1', 'l2']}
grid_lr = GridSearchCV(lr, param_grid = grid_values)#, scoring = 'roc_auc')
grid_lr = grid_lr.fit(X_train, y_train)

y_pred = grid_lr.predict(X_test)
print('Test Accuracy Score for LogisticRegression : ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for LogisticRegression : ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for LogisticRegression: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for LogisticRegression: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for LogisticRegression: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_lr.predict(X_train)
print('Train Accuracy Score for LogisticRegression: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for LogisticRegression: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for LogisticRegression: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for LogisticRegression: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for LogisticRegression: ', metrics.roc_auc_score(y_train, y_train_pred))
# XGBoost Model
xgb = XGBClassifier()

grid_values = {'max_depth': [8, 10, 15]}
grid_xgb = GridSearchCV(xgb, param_grid = grid_values)#, scoring = 'roc_auc')
grid_xgb = grid_xgb.fit(X_train, y_train)

y_pred = grid_xgb.predict(X_test)
print('Test Accuracy Score for XGBoost: ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for XGBoost: ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for XGBoost: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for XGBoost: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for XGBoost: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_xgb.predict(X_train)
print('Train Accuracy Score for XGBoost: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for XGBoost: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for XGBoost: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for XGBoost: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for XGBoost: ', metrics.roc_auc_score(y_train, y_train_pred))
from sklearn.svm import SVC
svc = SVC()

grid_values = {'C':[0.01,1], 'kernel': ['poly', 'rbf']}
grid_svc = GridSearchCV(svc, param_grid = grid_values)#, scoring = 'roc_auc')
grid_svc = grid_svc.fit(X_train, y_train)

y_pred = grid_svc.predict(X_test)
print('Test Accuracy Score for SVM: ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for SVM: ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for SVM: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for SVM: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for SVM: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_svc.predict(X_train)
print('Train Accuracy Score for SVM: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for SVM: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for SVM: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for SVM: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for SVM: ', metrics.roc_auc_score(y_train, y_train_pred))
# gbt = GradientBoostingClassifier()

# grid_values = {'learning_rate': [0.1, 1, 10],'n_estimators':[50, 100, 150],
#               'max_depth': [8]}
# grid_gbt = GridSearchCV(gbt, param_grid = grid_values, scoring = 'roc_auc')
# grid_gbt.fit(X_train, y_train)

# y_pred = grid_gbt.predict(X_test)
# print('Accuracy Score : ', metrics.accuracy_score(y_test, y_pred))
# print('Precision Score : ', metrics.precision_score(y_test, y_pred))
# print('Recall Score : ', metrics.recall_score(y_test, y_pred))
# print('F1 Score : ', metrics.f1_score(y_test, y_pred))
# print('auc Score : ', metrics.roc_auc_score(y_test, y_pred))
from imblearn.over_sampling import SMOTE
os_X_train, os_y_train = SMOTE().fit_resample(X_train, y_train)
# Running logistic regression model
lr = LogisticRegression()

grid_values = {'C':[0.001,0.01,1,10]}#, 'penalty': ['l1', 'l2']}
grid_lr = GridSearchCV(lr, param_grid = grid_values)#, scoring = 'roc_auc')
grid_lr = grid_lr.fit(os_X_train, os_y_train)

y_pred = grid_lr.predict(X_test)
print('Test Accuracy Score for LogisticRegression : ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for LogisticRegression : ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for LogisticRegression: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for LogisticRegression: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for LogisticRegression: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_lr.predict(X_train)
print('Train Accuracy Score for LogisticRegression: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for LogisticRegression: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for LogisticRegression: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for LogisticRegression: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for LogisticRegression: ', metrics.roc_auc_score(y_train, y_train_pred))
# XGBoost Model
xgb = XGBClassifier()

grid_values = {'max_depth': [8, 10, 15]}
grid_xgb = GridSearchCV(xgb, param_grid = grid_values)#, scoring = 'roc_auc')
grid_xgb = grid_xgb.fit(os_X_train, os_y_train)

y_pred = grid_xgb.predict(X_test)
print('Test Accuracy Score for XGBoost: ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for XGBoost: ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for XGBoost: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for XGBoost: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for XGBoost: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_xgb.predict(X_train)
print('Train Accuracy Score for XGBoost: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for XGBoost: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for XGBoost: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for XGBoost: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for XGBoost: ', metrics.roc_auc_score(y_train, y_train_pred))
from sklearn.svm import SVC
svc = SVC()

grid_values = {'C':[0.01,1], 'kernel': ['poly', 'rbf']}
grid_svc = GridSearchCV(svc, param_grid = grid_values)#, scoring = 'roc_auc')
grid_svc = grid_svc.fit(os_X_train, os_y_train)

y_pred = grid_svc.predict(X_test)
print('Test Accuracy Score for SVM: ', metrics.accuracy_score(y_test, y_pred))
print('Test Precision Score for SVM: ', metrics.precision_score(y_test, y_pred))
print('Test Recall Score for SVM: ', metrics.recall_score(y_test, y_pred))
print('Test F1 Score for SVM: ', metrics.f1_score(y_test, y_pred))
print('Test auc Score for SVM: ', metrics.roc_auc_score(y_test, y_pred))
print()
y_train_pred = grid_svc.predict(X_train)
print('Train Accuracy Score for SVM: ', metrics.accuracy_score(y_train, y_train_pred))
print('Train Precision Score for SVM: ', metrics.precision_score(y_train, y_train_pred))
print('Train Recall Score for SVM: ', metrics.recall_score(y_train, y_train_pred))
print('Train F1 Score for SVM: ', metrics.f1_score(y_train, y_train_pred))
print('Train auc Score for SVM: ', metrics.roc_auc_score(y_train, y_train_pred))
