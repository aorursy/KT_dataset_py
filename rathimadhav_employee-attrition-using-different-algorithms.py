# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.metrics import mean_absolute_error,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("/kaggle/input/employee-attrition/WA_Fn-UseC_-HR-Employee-Attrition.csv",usecols = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',

       'DistanceFromHome', 'Education', 'EducationField','EnvironmentSatisfaction',

       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',

       'MonthlyIncome', 'MonthlyRate', 'OverTime', 'PercentSalaryHike',

       'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',

       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',

       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',

       'YearsWithCurrManager'])

#here, we deleted columns that deos not matter or have the same values for all such as 

#Employee count, Standard Hours and to remove bias from our model, we've deleted some columns such as gender, marital status,

df = pd.get_dummies( df , columns=['BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction','JobInvolvement',

                                  'JobLevel','JobRole','JobSatisfaction','RelationshipSatisfaction',

                                  'WorkLifeBalance'] )

df.columns
df.drop(['BusinessTravel_Non-Travel','Department_Human Resources','Education_5','EducationField_Other','EnvironmentSatisfaction_4',

         'JobInvolvement_4','JobLevel_5','JobRole_Sales Representative','JobSatisfaction_4',

         'RelationshipSatisfaction_4','WorkLifeBalance_4'], axis=1, inplace=True) 
cleanup_cols = { 'Attrition' : {'Yes':1, 'No':0},

                'OverTime': {'Yes':1 , 'No':0},

                'PerformanceRating' : {3:0,4:1}

               }

#here we are not using dummies to create cols for these, instead we're directling replacing the values

# as we have only two values in each of three, we can directly replace them by 0 or 1

df.replace(cleanup_cols, inplace=True)
df.columns
y = df['Attrition']

X= df.loc[:, df.columns != 'Attrition']
from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier()

model.fit(X,y)

#print(model.feature_importances_)



#use inbuilt class feature_importances of tree based classifiers

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

#print(feat_importances<0.05)

feat_importances
#splitting the processed data for Model training and evaluation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#BASELINE MODEL WE WANT TO DEFEAT

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent") #Always predicts the most frequent class

dummy_clf.fit(X, y)

dummy_clf.predict(X)

dummy_clf.score(X, y) #Accuracy of the model that always predicts 0, i.e NO Attrition)
#RANDOM FOREST MODEL

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000,random_state = 0,max_depth = 20)



rf.fit(X_train, y_train)



y_pred = np.round(rf.predict(X_test)).astype(int)

accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

confusion_matrix(y_test, y_pred) #for Random Forest
#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(max_iter = 500,n_jobs=8)

reg.fit(X_train, y_train)



preds = reg.predict(X_test)

accuracy_score(y_test,preds)

accuracy = accuracy_score(y_test, preds, normalize=True, sample_weight=None)

accuracy
print(classification_report(y_test, preds))

confusion_matrix(y_test, preds) # for logistic regression
#Linear Discriminant Analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



clf = LinearDiscriminantAnalysis()

clf.fit(X_train,y_train)



prediction = clf.predict(X_test)



accuracy_score(y_test, prediction, normalize=True, sample_weight=None)
print(classification_report(y_test, prediction))

confusion_matrix(y_test, prediction) # for Linear Discriminant Analysis