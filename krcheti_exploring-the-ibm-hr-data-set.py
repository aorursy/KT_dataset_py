import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
%matplotlib inline
df = pd.read_csv('../input/HR-Em.csv')
df.head()
df.isnull().sum()
df.info()
mar = pd.get_dummies(df['MaritalStatus'])

o18 = pd.get_dummies(df['Over18'],'18')

gender = pd.get_dummies(df['Gender'],'sex')

edfield = pd.get_dummies(df['EducationField'],'field')

role = pd.get_dummies(df['JobRole'],'role')

travel = pd.get_dummies(df['BusinessTravel'],'travel')

department = pd.get_dummies(df['Department'],'dept')

attrition = pd.get_dummies(df['Attrition'],'attrition')

df = pd.concat([df,mar,o18,gender,edfield,role,travel,department,attrition],axis=1)
plt.figure(figsize=(20,5))

plt.bar(df['JobRole'],df['YearsAtCompany'],color='red',)
plt.xlabel('Job Role')
plt.ylabel('Years at company')
plt.figure(figsize=(20,5))

plt.bar(df['JobLevel'],df['YearsAtCompany'],color='green')
plt.ylabel('Years at company')
plt.xlabel('Job Level')
plt.figure(figsize=(20,5))

plt.bar(df['MonthlyIncome'],df['YearsSinceLastPromotion'],color='m')

plt.figure(figsize=(10,10))
plt.scatter(df['Age'],df['YearsAtCompany'],color='k', s=2)
plt.xlabel('AGE')
plt.ylabel('YearsAtCompany')
sns.factorplot(x= 'MaritalStatus',data=df,
             hue='BusinessTravel',
             col='Gender',
             kind='count');
sns.boxplot(x='BusinessTravel', y='DailyRate',
           hue='MaritalStatus', palette=['m', 'g', 'r'],
           data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
ndf = df.select_dtypes(include=['int64','uint8'])
ndf.columns
X = ndf[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Divorced',
       'Married', 'Single', '18_Y', 'sex_Female', 'sex_Male',
       'field_Human Resources', 'field_Life Sciences', 'field_Marketing',
       'field_Medical', 'field_Other', 'field_Technical Degree',
       'role_Healthcare Representative', 'role_Human Resources',
       'role_Laboratory Technician', 'role_Manager',
       'role_Manufacturing Director', 'role_Research Director',
       'role_Research Scientist', 'role_Sales Executive',
       'role_Sales Representative', 'travel_Non-Travel',
       'travel_Travel_Frequently', 'travel_Travel_Rarely',
       'dept_Human Resources', 'dept_Research & Development', 'dept_Sales',
       'Divorced', 'Married', 'Single', '18_Y', 'sex_Female', 'sex_Male',
       'field_Human Resources', 'field_Life Sciences', 'field_Marketing',
       'field_Medical', 'field_Other', 'field_Technical Degree',
       'role_Healthcare Representative', 'role_Human Resources',
       'role_Laboratory Technician', 'role_Manager',
       'role_Manufacturing Director', 'role_Research Director',
       'role_Research Scientist', 'role_Sales Executive',
       'role_Sales Representative', 'travel_Non-Travel',
       'travel_Travel_Frequently', 'travel_Travel_Rarely',
       'dept_Human Resources', 'dept_Research & Development', 'dept_Sales']]
y = df['attrition_Yes']
X_train,X_test,y_train,y_test = train_test_split(X,y)
tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train,y_train)
tt = pd.DataFrame([tree.predict(X_test),y_test],index=['prediction','actual']).T
tt[tt['prediction'] == tt['actual']].shape[0] / tt.shape[0]
1 - tt['actual'].mean()

knn = KNeighborsClassifier()
params_grid = {
    'n_neighbors':[21,23,25],
    'algorithm':['auto','brute','kd_tree','ball_tree'],
    'p':[2]
}
gs = GridSearchCV(KNeighborsClassifier(),params_grid,verbose=3)
gs.fit(X_train,y_train)
gs.best_params_
kp = pd.DataFrame([gs.predict(X_test),y_test],['preds','actuals']).T
kp[kp['preds']==kp['actuals']].shape[0] / kp.shape[0]  
X = ndf.drop(['Divorced','Married','Single'],axis=1)
y = ndf['Divorced']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))