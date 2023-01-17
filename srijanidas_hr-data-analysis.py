import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
employee_survey = pd.read_csv("../input/hr-analytics-case-study/employee_survey_data.csv")
manager_survey = pd.read_csv("../input/hr-analytics-case-study/manager_survey_data.csv")
general_data = pd.read_csv("../input/hr-analytics-case-study/general_data.csv")
print(employee_survey.columns)
print(manager_survey.columns)
print(general_data.columns)
from functools import reduce
df_list = [employee_survey, manager_survey, general_data]
emp_df = reduce(lambda left,right: pd.merge(left,right,how='inner',on='EmployeeID'), df_list)
emp_df.columns
emp_df.shape
emp_df.info()
emp_df.describe()
print(emp_df['Over18'].unique())
print(emp_df['EmployeeCount'].unique())
print(emp_df['StandardHours'].unique())
# This function takes the dataframe and list of features to be dropped
# returns the updated dataframe

def drop_features(df, feat_list):
    for col in feat_list:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"{col} is dropped")
        else:
            print(f"{col} is already dropped")
drop_features(emp_df, ['EmployeeID', 'EmployeeCount', 'Over18', 'StandardHours'])
print("The categorical columns and their index-")
for col in emp_df.columns:
    if emp_df[col].dtype == 'object':
        print(col, emp_df.columns.get_loc(col))
def show_percentage_of_people_left(column_name):
    df = emp_df.groupby(column_name)['Attrition'].describe()
    df['percentage of people left'] = (1 - (df['freq']/df['count']))*100
    print(df)
    print('===============================')
for col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']:
    show_percentage_of_people_left(col)
sns.countplot(x='Attrition', hue='BusinessTravel', data=emp_df)
sns.countplot(x='Attrition', hue='Department', data=emp_df)
sns.countplot(x='Attrition', hue='EducationField', data=emp_df)
sns.countplot(x='Attrition', hue='Gender', data=emp_df)
sns.countplot(x='Attrition', hue='JobRole', data=emp_df)
sns.countplot(x='Attrition', hue='MaritalStatus', data=emp_df)
emp_df['Single'] = pd.get_dummies(emp_df["MaritalStatus"])['Single']
pd.set_option('mode.chained_assignment', None)

emp_df['RD'] = np.zeros(emp_df.shape[0])
emp_df['LT_RS_SE'] = np.zeros(emp_df.shape[0])

for row_num in range(0, emp_df.shape[0]):
    if emp_df['JobRole'][row_num] == 'Research Director':
        emp_df['RD'][row_num] = 1
    if emp_df['JobRole'][row_num] in ['Laboratory Technician', 'Research Scientist', 'Sales Executive']:
        emp_df['LT_RS_SE'][row_num] = 1
emp_df['Male'] = pd.get_dummies(emp_df["Gender"])["Male"]

# In EducationField
# HR : Avg Attrition Rate 40 %
# Others : Avg Attrition Rate 14 %
emp_df["EducationField_HR"] = pd.get_dummies(emp_df["EducationField"], prefix='EducationField')["EducationField_Human Resources"]

# In Department
# HR : Avg Attrition Rate 30 %
# Others : Avg Attrition Rate 15 %

emp_df["Department_HR"] = pd.get_dummies(emp_df["Department"], prefix='Department')["Department_Human Resources"]

emp_df["Travel_Frequently"] = pd.get_dummies(emp_df["BusinessTravel"])["Travel_Frequently"]
emp_df["Travel_Rarely"] = pd.get_dummies(emp_df["BusinessTravel"])["Travel_Rarely"]

emp_df["Attrition_Yes"] = pd.get_dummies(emp_df["Attrition"], prefix='Attrition')["Attrition_Yes"]
drop_features(emp_df, ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Attrition'])
emp_df.shape
emp_df.info()
plt.figure(figsize=(24,10))
sns.heatmap(emp_df.corr(), annot=True)
drop_features(emp_df, ["DistanceFromHome", "StockOptionLevel", "PercentSalaryHike", "YearsSinceLastPromotion", "YearsWithCurrManager", "TotalWorkingYears"])
emp_df.shape
emp_df.columns
sns.jointplot(emp_df['Age'], emp_df['NumCompaniesWorked'], data=emp_df, kind='kde')
sns.jointplot(emp_df['Age'], emp_df['NumCompaniesWorked'], data=emp_df, kind='hex')
sns.jointplot(emp_df['JobLevel'], emp_df['NumCompaniesWorked'], data=emp_df, kind='kde')
sns.countplot(emp_df['EnvironmentSatisfaction'], hue=emp_df['Attrition_Yes'], data=emp_df)
sns.countplot(emp_df['JobSatisfaction'], hue=emp_df['Attrition_Yes'], data=emp_df)
sns.countplot(emp_df['WorkLifeBalance'], hue=emp_df['Attrition_Yes'], data=emp_df)
plt.figure(figsize=(18,10))
sns.countplot(emp_df['Age'], hue=emp_df['Attrition_Yes'], data=emp_df)
X = emp_df.iloc[:, :-1].values
y = emp_df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
plt.figure(figsize=(24,10))
sns.heatmap(emp_df.isnull())
for col in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'NumCompaniesWorked']:
  print("Column : ", col)
  print("Mean : ", emp_df[col].mean())
  print("Mode : ", emp_df[col].mode())
  print("Unique values : ", emp_df[col].unique())
  print("Index : ", emp_df.columns.get_loc(col))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_const = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=3)

# for 'EnvironmentSatisfaction', 'WorkLifeBalance', 'NumCompaniesWorked'

for col in [0, 2, 9]:
    imputer.fit(X_train[:, col].reshape(X_train[:, col].shape[0], 1))
    X_train[:, col] = imputer.transform(X_train[:, col].reshape(X_train[:, col].shape[0], 1))[:, 0]
    X_test[:, col] = imputer.transform(X_test[:, col].reshape(X_test[:, col].shape[0], 1))[:, 0]

# for 'JobSatisfaction'
col = 1
imputer_const.fit(X_train[:, col].reshape(X_train[:, col].shape[0], 1))
X_train[:, col] = imputer_const.transform(X_train[:, col].reshape(X_train[:, col].shape[0], 1))[:, 0]
X_test[:, col] = imputer_const.transform(X_test[:, col].reshape(X_test[:, col].shape[0], 1))[:, 0]
for i in [0, 1, 2, 9]:
    array_sum = np.sum(X_train[:,i])
    array_has_nan = np.isnan(array_sum)
    print(array_has_nan)
    
    array_sum = np.sum(X_test[:,i])
    array_has_nan = np.isnan(array_sum)
    print(array_has_nan)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
