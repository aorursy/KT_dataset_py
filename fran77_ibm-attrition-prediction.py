# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()
data.columns
plt.figure(figsize=(15,8))

ax = sns.countplot(data.Age)
plt.figure(figsize=(8,5))

ax = sns.countplot(data.Attrition)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.BusinessTravel) 
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.DailyRate)
plt.figure(figsize=(10,6))



ax = sns.kdeplot(data.DistanceFromHome)
# Right Skew
plt.figure(figsize=(10,6))

ax = sns.countplot(data.Department) 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.Education) 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.EducationField) 
data.EmployeeCount.value_counts()
# Not useful
# Not useful 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.EnvironmentSatisfaction)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.Gender) 
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.HourlyRate)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.JobInvolvement)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.JobLevel)
plt.figure(figsize=(15,6))

ax = sns.countplot(data.JobRole)

t=plt.xticks(rotation=45) 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.JobSatisfaction)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.MaritalStatus) 
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.MonthlyIncome)
# Right Skew
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.MonthlyRate)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.NumCompaniesWorked)
data.Over18.value_counts()
# Not useful 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.OverTime) 
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.PercentSalaryHike) 
# A little right skew
plt.figure(figsize=(10,6))

ax = sns.countplot(data.PerformanceRating)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.RelationshipSatisfaction)
data.StandardHours.value_counts()
# Not useful 
plt.figure(figsize=(10,6))

ax = sns.countplot(data.StockOptionLevel)
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.TotalWorkingYears) 
# A little right skew
plt.figure(figsize=(10,6))

ax = sns.countplot(data.TrainingTimesLastYear)
plt.figure(figsize=(10,6))

ax = sns.countplot(data.WorkLifeBalance)
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.YearsAtCompany) 
# Right skew
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.YearsInCurrentRole)
# There are 2 curves : young people vs older people
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.YearsSinceLastPromotion) 
# Right skew
plt.figure(figsize=(10,6))

ax = sns.kdeplot(data.YearsWithCurrManager)
# Almost the same as YearsInCurrentRole
g=data.groupby('Age')['YearsAtCompany'].mean().plot()
g=data.groupby('Age')['YearsInCurrentRole'].mean().plot()
# Normalize features columns

# Models performe better when values are close to normally distributed

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
data['DistanceFromHome'] = scaler.fit_transform(data['DistanceFromHome'].values.reshape(-1, 1))

data['MonthlyIncome'] = scaler.fit_transform(data['MonthlyIncome'].values.reshape(-1, 1))

data['PercentSalaryHike'] = scaler.fit_transform(data['PercentSalaryHike'].values.reshape(-1, 1))

data['TotalWorkingYears'] = scaler.fit_transform(data['TotalWorkingYears'].values.reshape(-1, 1))

data['YearsAtCompany'] = scaler.fit_transform(data['YearsAtCompany'].values.reshape(-1, 1))

data['YearsSinceLastPromotion'] = scaler.fit_transform(data['YearsSinceLastPromotion'].values.reshape(-1, 1))
# Convert to categorical values

data['Attrition'] = data.Attrition.astype('category').cat.codes

data['BusinessTravel'] = data.BusinessTravel.astype('category').cat.codes

data['Department'] = data.Department.astype('category').cat.codes

data['EducationField'] = data.EducationField.astype('category').cat.codes

data['Gender'] = data.Gender.astype('category').cat.codes

data['JobRole'] = data.JobRole.astype('category').cat.codes

data['MaritalStatus'] = data.MaritalStatus.astype('category').cat.codes 

data['OverTime'] = data.OverTime.astype('category').cat.codes
# Check NA

data.isnull().sum(axis = 0)
# Remove columns not useful

data = data.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)
# Get columns with at least 0.1 correlation

data_corr = data.corr()['Attrition'] # Attrition : column to predict

cols = data_corr[abs(data_corr) > 0.1].index.tolist()

data = data[cols]
# plot the heatmap

data_corr = data.corr()

plt.figure(figsize=(10,8))

sns.heatmap(data_corr, 

        xticklabels=data_corr.columns,

        yticklabels=data_corr.columns, cmap=sns.diverging_palette(220, 20, n=200))
data.corr()['Attrition'].sort_values(ascending=False)
# Check correlations between columns

data['JobLevel'].corr(data['MonthlyIncome'])
# Too much correlation
data['YearsAtCompany'].corr(data['YearsWithCurrManager'])
data['JobLevel'].corr(data['TotalWorkingYears'])
data['YearsInCurrentRole'].corr(data['YearsWithCurrManager'])
# Remove columns with too much correlation

data = data.drop(["MonthlyIncome", "YearsAtCompany", "JobLevel", "YearsWithCurrManager"], axis=1)
data.columns
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
X = data.drop("Attrition", axis=1)

Y = data["Attrition"]
# Split 20% test, 80% train



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=1000)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_test)

acc_log = accuracy_score(Y_pred_log, Y_test)

acc_log
t = tree.DecisionTreeClassifier()



# search the best params

grid = {'min_samples_split': [5, 10, 20, 50, 100]},



clf_tree = GridSearchCV(t, grid, cv=10)

clf_tree.fit(X_train, Y_train)



Y_pred_tree = clf_tree.predict(X_test)



# get the accuracy score

acc_tree = accuracy_score(Y_pred_tree, Y_test)

print(acc_tree)
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': [2,5,10]}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_test)

print(acc_rf)
# The best model is Logistic Regression