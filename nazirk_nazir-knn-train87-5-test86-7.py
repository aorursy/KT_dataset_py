# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
empl_df = pd.read_csv("../input/HR-Employee-Attrition.csv")
empl_df.isna().sum()
cols_to_Encode = ["BusinessTravel","Department","EducationField","JobRole","MaritalStatus","Over18","OverTime"]

numeric_cols = ["Attrition", "Gender","Age","DailyRate","DistanceFromHome","Education","EmployeeCount","EmployeeNumber","EnvironmentSatisfaction","HourlyRate",

                "JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike",

               "PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear"

                ,"WorkLifeBalance","YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion","YearsWithCurrManager"]
empl_df['Attrition'] = empl_df['Attrition'].apply(lambda x: 1 if x=="Yes" else 0)

empl_df['Attrition'].value_counts()
empl_df['Gender'] = empl_df['Gender'].apply(lambda x: 1 if x=="Male" else 0)

empl_df['Gender'].value_counts()
encoded_cols = pd.get_dummies(empl_df[cols_to_Encode])
df_final = pd.concat([encoded_cols,empl_df[numeric_cols]], axis = 1)
df_final.shape
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
X=df_final.drop(columns=['Attrition'])

y=df_final['Attrition']
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
k = 5

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)
df_final.head()
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
metrics.confusion_matrix(y_train, neigh.predict(X_train))
metrics.confusion_matrix(y_test, yhat)