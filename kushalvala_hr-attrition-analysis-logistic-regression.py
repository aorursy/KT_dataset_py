import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'], axis=1, inplace=True)
df.head()
df.columns
df = pd.get_dummies( df , columns=['BusinessTravel','Department','Education','EducationField','EnvironmentSatisfaction','Gender','JobInvolvement',
                                  'JobLevel','JobRole','JobSatisfaction','MaritalStatus','PerformanceRating','RelationshipSatisfaction',
                                  'WorkLifeBalance'] )
df.shape
df.head()
df.drop(['BusinessTravel_Non-Travel','Department_Human Resources','Education_5','EducationField_Other','EnvironmentSatisfaction_4','Gender_Female'
        ,'JobInvolvement_4','JobLevel_5','JobRole_Sales Representative','JobSatisfaction_4','MaritalStatus_Divorced','PerformanceRating_3',
        'RelationshipSatisfaction_4','WorkLifeBalance_4'], axis=1, inplace=True)
df.shape
#Doing a Manual Clean-up Encoding on 'Attrition' , 'OverTime'

cleanup_nums = { 'Attrition' : {'Yes':1, 'No':0},
                'OverTime': {'Yes':1 , 'No':0}
               }

df.replace(cleanup_nums, inplace=True)
df.head()
df.columns
X = df.loc[:, ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Research & Development', 'Department_Sales', 'Education_1',
       'Education_2', 'Education_3', 'Education_4',
       'EducationField_Human Resources', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Technical Degree', 'EnvironmentSatisfaction_1',
       'EnvironmentSatisfaction_2', 'EnvironmentSatisfaction_3', 'Gender_Male',
       'JobInvolvement_1', 'JobInvolvement_2', 'JobInvolvement_3',
       'JobLevel_1', 'JobLevel_2', 'JobLevel_3', 'JobLevel_4',
       'JobRole_Healthcare Representative', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Research Scientist', 'JobRole_Sales Executive',
       'JobSatisfaction_1', 'JobSatisfaction_2', 'JobSatisfaction_3',
       'MaritalStatus_Married', 'MaritalStatus_Single', 'PerformanceRating_4',
       'RelationshipSatisfaction_1', 'RelationshipSatisfaction_2',
       'RelationshipSatisfaction_3', 'WorkLifeBalance_1', 'WorkLifeBalance_2',
       'WorkLifeBalance_3']]


y = df.loc[:, 'Attrition']
print(X.shape)
print(y.shape)
df.corr()
X.columns
X.drop(['HourlyRate','Education_2','JobRole_Research Scientist','JobSatisfaction_2','JobSatisfaction_3','PerformanceRating_4'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=114)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
reg.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)