import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/HR-Employee-Attrition.csv', header=0)

data = data.dropna()

print(data.shape)
print(list(data.columns))
data.head()
data.drop(columns='Attrition').dtypes
print(data['Attrition'].dtype)
data.isna().sum()
data.duplicated().sum()
data['Attrition'].replace({'No':0,'Yes':1},inplace=True)
num_cols = data.select_dtypes(include = np.number)
a = num_cols[num_cols.columns].hist(bins=15, figsize=(15,35), layout=(9,3),color = 'red',alpha=0.6)
cat_col = data.select_dtypes(exclude=np.number)
cat_col.columns
fig, ax = plt.subplots(4, 2, figsize=(15, 15))

for variable, subplot in zip(cat_col, ax.flatten()):

    sns.countplot(data[variable], ax=subplot,palette = 'plasma')

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

plt.tight_layout()
data[['StandardHours','EmployeeCount']].describe()
data[['StandardHours','EmployeeCount']].corr()
corr = data.drop(columns=['StandardHours','EmployeeCount']).corr()

corr.style.background_gradient(cmap='YlGnBu')
cols = ['Age', 'BusinessTravel', 'Department',

       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',

        'EnvironmentSatisfaction', 'Gender', 

       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',

       'MaritalStatus', 'NumCompaniesWorked',

       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager']

for col in cols:

    pd.crosstab(data[col],data.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(10,5))
# Age Vs Attrition - From data, it appears that attrition is more at age group 18-23

# % of attrition is more among people who travel frequently

# % of attrition is more in sales department

# %of attrition is more during 0-1 years of working in company

# People in job role of Sales Representative tend to have more attrition %

# From given data, overtime population has more attrition
data.columns.shape
cat_col.columns.shape
num_cols.columns.shape
cat_col_encoded = pd.get_dummies(cat_col)
cat_col_encoded.head()
df = pd.concat([num_cols,cat_col_encoded],sort=False,axis=1)
df.head()
X = df.drop(columns='Attrition')
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_Pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.accuracy_score(y_train,train_Pred)
test_Pred = logreg.predict(X_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()