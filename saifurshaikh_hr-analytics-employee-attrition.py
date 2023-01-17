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
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.shape
df.info()
df.describe()
for col in df.columns:
    if 'Satisfaction' in col:
        print(col)
for col in df.columns:
    if 'Satisfaction' in col:
        print(df[col].describe())
import seaborn as sns
import matplotlib.pyplot as plt
df['OverallSatisfaction'] = (df['EnvironmentSatisfaction']+df['JobSatisfaction']+df['RelationshipSatisfaction'])/3
df.head()
df.drop(['EnvironmentSatisfaction','JobSatisfaction','RelationshipSatisfaction'], axis=1, inplace=True)
df.head()
### Dropping Employee count column as it has only value 1 throughout
### Also, dropping employee number column as it is of no use as well (same as Emp Id)
df.drop(['EmployeeCount','EmployeeNumber'], axis=1, inplace=True)
df.describe()
df['AgeGroup'] = 'Old'
df.loc[df['Age']<=30, 'AgeGroup'] = 'Young'
df.loc[(df['Age']>30) & (df['Age']<=50), 'AgeGroup'] = 'MidAge'
df.head()
df.drop('Age', axis=1, inplace=True)
df.head()
df.describe()
df.corr()
### We see that there is almost no multicollinearity between numerical columns
### Also, column StandardHours have 8 throughout. So, dropping this column as well
df.drop('StandardHours', axis=1, inplace=True)
sns.distplot(df.MonthlyRate)
sns.distplot(df.HourlyRate)
df[['MonthlyRate','HourlyRate','MonthlyIncome']]
plt.figure(figsize=(20,20))
plt.scatter(x='MonthlyRate', y='HourlyRate', data=df)
### From above, we see that either MonthlyRate or HourlyRate column is required
### As we have MonthlyIncome column as well, we don't need MonthlyRate anymore
df.drop('MonthlyRate', axis=1, inplace=True)
df.head()
df.Attrition.value_counts()
df.Attrition = np.where(df.Attrition=='Yes',1,0)
df.Attrition = pd.to_numeric(df.Attrition, errors='coerce')
df.info()
df_num = df.select_dtypes('number')
df_num.head()
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(score_func=chi2, k=10).fit(df_num.iloc[:,1:], df_num.Attrition)
kbest = selector.transform(df_num.iloc[:,1:])
print(kbest.shape)
df_num.iloc[:,1:].columns[selector.get_support(indices=True)]
df_num_kbest = df_num[['DailyRate', 'DistanceFromHome', 'JobLevel', 'MonthlyIncome',
       'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager','Attrition']]
df_num_kbest.head()
df_char = df.select_dtypes('object')
df_all = pd.concat([df_char, df_num_kbest], axis=1)
df_all.head()
df_all.shape
df_all.Over18.value_counts()
### There is only 1 value 'Y' in this columns. So, dropping this
df_all.drop('Over18', axis=1, inplace=True)
df_all.head()
df_final = pd.get_dummies(data=df_all, drop_first=True)
df_final.shape
df_final.head()
X = df_final.drop('Attrition', axis=1)
y = df_final.Attrition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify=y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.transform(X_test)
X_train_scaled1 = pd.DataFrame(X_train_scaled, columns=X_train.columns.values)
X_test_scaled1 = pd.DataFrame(y_train_scaled, columns=X_test.columns.values)
et_clf = ExtraTreeClassifier(random_state=2)
et_clf.fit(X_train_scaled1, y_train)
y_pred = et_clf.predict(X_test_scaled1)
acc = metrics.accuracy_score(y_test, y_pred)
acc
cf_mat = metrics.confusion_matrix(y_test, y_pred)
print(cf_mat)
et_clf.feature_importances_
features = pd.DataFrame(et_clf.feature_importances_, index=X_train_scaled1.columns, columns=['Score'])
features.nlargest(12, columns=['Score']).index
X_train_imp = X_train_scaled1[['DailyRate', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany',
       'OverTime_Yes', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'StockOptionLevel', 'MonthlyIncome', 'AgeGroup_Young',
       'YearsInCurrentRole', 'JobLevel']]
X_test_imp = X_test_scaled1[['DailyRate', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany',
       'OverTime_Yes', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'StockOptionLevel', 'MonthlyIncome', 'AgeGroup_Young',
       'YearsInCurrentRole', 'JobLevel']]
Name = "ExtraTreesClf"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_pred, y_test))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_pred, y_test))
svc = SVC(random_state=2, C=10)
svc.fit(X_train_imp, y_train)
y_pred_svc = svc.predict(X_test_imp)
Name = "SVC"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_pred_svc, y_test))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_pred_svc, y_test))
Log_reg = LogisticRegression(penalty='none')
Log_reg.fit(X_train_imp, y_train)
y_log_pred = Log_reg.predict(X_test_imp)
Name = "Log Reg"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_log_pred, y_test))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_log_pred, y_test))
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train_imp, y_train)
y_xgb_pred = xgb_clf.predict(X_test_imp)
Name = "XGBoost"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_xgb_pred, y_test))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_xgb_pred, y_test))
rf_clf = RandomForestClassifier(random_state=2, max_features=5, max_depth=5)
rf_clf.fit(X_train_imp, y_train)
y_rf_pred = rf_clf.predict(X_test_imp)
Name = "Random Forest"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_test, y_rf_pred))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_test, y_rf_pred))
rf_clf.get_params
params = {'n_estimators' : [100, 300, 500, 700, 1000],
         'max_depth' : [3,4,5,6,7,9,10],
         'min_samples_split' : [10,15,20,25,30],
         'min_samples_leaf' : [10,15,20,25,30],
         'max_leaf_nodes' : [10,15,20,25,30]}
rf_clf = RandomForestClassifier(random_state=2)
cv_clf = RandomizedSearchCV(rf_clf, param_distributions=params, cv=5, scoring='accuracy', verbose=1)
cv_clf.fit(X_train_imp, y_train)
cv_clf.best_estimator_
rf_clf = RandomForestClassifier(max_depth=9, max_leaf_nodes=25, min_samples_leaf=10,
                       min_samples_split=25, n_estimators=700, random_state=2)
rf_clf.fit(X_train_imp, y_train)
y_rf_pred = rf_clf.predict(X_test_imp)
Name = "Random Forest"
print(Name)
print('Accuracy: ',metrics.accuracy_score(y_test, y_rf_pred))
print('ROC_AUC_Score: ',metrics.roc_auc_score(y_test, y_rf_pred))
### If we check Accuracy and ROC_AUC both, ExtraTreeClassifier and Log Regression has performed better
### Log Reg
### Accuracy:  0.832579185520362
### ROC_AUC_Score:  0.6534632034632034

### ExtraTreesClf
### Accuracy:  0.8190045248868778
### ROC_AUC_Score:  0.6521095484826055
