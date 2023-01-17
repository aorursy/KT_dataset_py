# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tpot import TPOTClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



df_train = pd.read_csv('../input/rs6-attrition-predict/train.csv')

# int64(27), object(9) 1176 entries in total

print(df_train.info())



print(df_train.head())



df_train['Attrition'].describe()



# to test whether NaN values inside 

df_train.isnull().sum().sort_values().head()
# df_train.index.names

df_train.columns
df_train['Attrition'].unique()

df_train["Attrition"] = pd.factorize(df_train["Attrition"])[0].astype(np.int64)

df_train['Attrition'].unique()

# array(['No', 'Yes'], dtype=object)

# array([0, 1], dtype=uint64)



df_train['BusinessTravel'].unique()

df_train["BusinessTravel"] = pd.factorize(df_train["BusinessTravel"])[0].astype(np.int64)

df_train['BusinessTravel'].unique()

# array(['Travel_Rarely', 'Non-Travel', 'Travel_Frequently'], dtype=object)

# array([0, 1, 2], dtype=uint64)



# Department

df_train['Department'].unique()

df_train["Department"] = pd.factorize(df_train["Department"])[0].astype(np.int64)

df_train['Department'].unique()

# array(['Sales', 'Research & Development', 'Human Resources'], dtype=object)

# array([0, 1, 2], dtype=uint64)



# EducationField

df_train['EducationField'].unique()

df_train["EducationField"] = pd.factorize(df_train["EducationField"])[0].astype(np.int64)

df_train['EducationField'].unique()

# array(['Life Sciences', 'Technical Degree', 'Marketing', 'Medical',

#        'Human Resources', 'Other'], dtype=object)

# array([0, 1, 2, 3, 4, 5], dtype=uint64)



# Gender

df_train['Gender'].unique()

df_train["Gender"] = pd.factorize(df_train["Gender"])[0].astype(np.int64)

df_train['Gender'].unique()

# array(['Female', 'Male'], dtype=object)

# array([0, 1], dtype=uint64)



# JobRole

df_train['JobRole'].unique()

df_train["JobRole"] = pd.factorize(df_train["JobRole"])[0].astype(np.int64)

df_train['JobRole'].unique()

# array(['Manager', 'Research Scientist', 'Sales Executive',

#        'Sales Representative', 'Laboratory Technician',

#        'Manufacturing Director', 'Human Resources', 'Research Director',

#        'Healthcare Representative'], dtype=object)

# array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=uint64)



# Over18, useless info

df_train['Over18'].unique()

df_train['Over18'].value_counts()



# OverTime

df_train['OverTime'].unique()

df_train['OverTime'].value_counts()

df_train['OverTime'].unique()

df_train["OverTime"] = pd.factorize(df_train["OverTime"])[0].astype(np.int64)

df_train['OverTime'].unique()

# array(['Yes', 'No'], dtype=object)

# array([0, 1], dtype=uint64)
# MaritalStatus

df_train['MaritalStatus'].unique()

df_train["MaritalStatus"] = pd.factorize(df_train["MaritalStatus"])[0].astype(np.int64)

df_train['MaritalStatus'].unique()

# array(['Married', 'Single', 'Divorced'], dtype=object)

# array([0, 1, 2])
cols = ['user_id', 'Age', 'BusinessTravel', 'DailyRate',

       'Department', 'DistanceFromHome', 'Education', 'EducationField',

       'EnvironmentSatisfaction', 'Gender',

       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',

       'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',

       'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',

       'PerformanceRating', 'RelationshipSatisfaction',

       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',

       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',

       'YearsSinceLastPromotion', 'YearsWithCurrManager']

df_train_re = DataFrame(df_train, columns = cols)

df_train_re.info()
x = df_train_re[cols].values

y = df_train['Attrition'].values

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 42)

# type(y_train)

X_train = np.asmatrix(X_train)

y_train = np.array(y_train)

X_test = np.asmatrix(X_test)

y_test = np.array(y_test)
'''

The key point of this assignment is to predict score... not to do the classification.

There the model acc is so low... and mainly cuz my data pre-processing is based on the need of classifier 

'''



# model = GradientBoostingClassifier(random_state=25)

model = RandomForestRegressor(n_estimators=60, random_state=None)

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
df_test = pd.read_csv('../input/rs6-attrition-predict/test.csv')

# df_train['Attrition'].unique()

# df_train["Attrition"] = pd.factorize(df_train["Attrition"])[0].astype(np.int64)

# df_train['Attrition'].unique()

# array(['No', 'Yes'], dtype=object)

# array([0, 1], dtype=uint64)



df_test['BusinessTravel'].unique()

df_test["BusinessTravel"] = pd.factorize(df_test["BusinessTravel"])[0].astype(np.int64)

df_test['BusinessTravel'].unique()

# array(['Travel_Rarely', 'Non-Travel', 'Travel_Frequently'], dtype=object)

# array([0, 1, 2], dtype=uint64)



# Department

df_test['Department'].unique()

df_test["Department"] = pd.factorize(df_test["Department"])[0].astype(np.int64)

df_test['Department'].unique()

# array(['Sales', 'Research & Development', 'Human Resources'], dtype=object)

# array([0, 1, 2], dtype=uint64)



# EducationField

df_test['EducationField'].unique()

df_test["EducationField"] = pd.factorize(df_test["EducationField"])[0].astype(np.int64)

df_test['EducationField'].unique()

# array(['Life Sciences', 'Technical Degree', 'Marketing', 'Medical',

#        'Human Resources', 'Other'], dtype=object)

# array([0, 1, 2, 3, 4, 5], dtype=uint64)



# Gender

df_test['Gender'].unique()

df_test["Gender"] = pd.factorize(df_test["Gender"])[0].astype(np.int64)

df_test['Gender'].unique()

# array(['Female', 'Male'], dtype=object)

# array([0, 1], dtype=uint64)



# JobRole

df_test['JobRole'].unique()

df_test["JobRole"] = pd.factorize(df_test["JobRole"])[0].astype(np.int64)

df_test['JobRole'].unique()

# array(['Manager', 'Research Scientist', 'Sales Executive',

#        'Sales Representative', 'Laboratory Technician',

#        'Manufacturing Director', 'Human Resources', 'Research Director',

#        'Healthcare Representative'], dtype=object)

# array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=uint64)



# Over18, useless info

df_test['Over18'].unique()

df_test['Over18'].value_counts()



# OverTime

df_test['OverTime'].unique()

df_test['OverTime'].value_counts()

df_test['OverTime'].unique()

df_test["OverTime"] = pd.factorize(df_test["OverTime"])[0].astype(np.int64)

df_test['OverTime'].unique()

# array(['Yes', 'No'], dtype=object)

# array([0, 1], dtype=uint64)



# MaritalStatus

df_test['MaritalStatus'].unique()

df_test["MaritalStatus"] = pd.factorize(df_test["MaritalStatus"])[0].astype(np.int64)

df_test['MaritalStatus'].unique()

# array(['Married', 'Single', 'Divorced'], dtype=object)

# array([0, 1, 2])
cols = ['user_id', 'Age', 'BusinessTravel', 'DailyRate',

       'Department', 'DistanceFromHome', 'Education', 'EducationField',

       'EnvironmentSatisfaction', 'Gender',

       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',

       'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',

       'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',

       'PerformanceRating', 'RelationshipSatisfaction',

       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',

       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',

       'YearsSinceLastPromotion', 'YearsWithCurrManager']

df_test_re = DataFrame(df_test, columns = cols)
X_test_pred = df_test_re[cols].values

# y_pred = df_test['Attrition'].values

# X_train_pred,X_test_pred,y_train_pred,y_test_pred = train_test_split(x_pred,y_pred,test_size = 0.2,random_state= 42)

# type(y_train)

X_test_pred = np.asmatrix(X_test_pred)

y_pred = model.predict(X_test_pred)

prediction = pd.DataFrame(y_pred, columns=['Attrition'])

result = pd.concat([df_test_re['user_id'], prediction], axis=1)

# result.to_csv('results2.csv', index=False)
result