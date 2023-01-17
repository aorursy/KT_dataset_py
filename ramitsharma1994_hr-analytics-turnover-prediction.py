import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

data = pd.read_csv('../input/hr-analytics/general_data.csv')
emp_survey = pd.read_csv("../input/hr-analytics/employee_survey_data.csv")
in_time = pd.read_csv("../input/hr-analytics/in_time.csv")
manager_survey = pd.read_csv("../input/hr-analytics/manager_survey_data.csv")
out_time = pd.read_csv("../input/hr-analytics/out_time.csv")
#data_dictionary = pd.read_excel("../input/hr-analytics/data_dictionary.xlsx")
diff_time = pd.read_csv("../input/hr-analytics/difference_time.csv")
emp_survey.head()
data = pd.merge(data,emp_survey, on = 'EmployeeID', how = 'outer')
data.head()
data = pd.merge(data,manager_survey, on = 'EmployeeID', how = 'outer')
data.head()
diff_time = diff_time.replace("#VALUE!", np.nan)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = df[feature_name].astype(float)
    return result

normalize(diff_time)

diff_time1 = diff_time.iloc[:,1:]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(diff_time1.T)
diff_time1 = imputer.transform(diff_time1.T)
diff_time2 = pd.DataFrame(diff_time1.T)
diff_time['Average_Time'] = diff_time2.mean(axis = 1)
diff_time = diff_time[['EmployeeID','Average_Time']]
data2 = pd.merge(data,diff_time, on = 'EmployeeID', how = 'outer')
data2.head()
final_data = data2.copy()

travel = {'Non-Travel': 0, "Travel_Rarely": 1, "Travel_Frequently":2}
gender = {"Male": 0, "Female": 1}
dept = {"Sales": 0, "Research & Development":1, "Human Resources": 2}
married = {"Single": 0, "Married":1, "Divorced": 2}
attrition = {"Yes":1, "No":0}

final_data['BusinessTravel'] = final_data['BusinessTravel'].map(lambda x: travel[x])
final_data['Gender'] = final_data['Gender'].map(lambda x: gender[x])
final_data['Department'] = final_data['Department'].map(lambda x: dept[x])
final_data['Attrition'] = final_data['Attrition'].map(lambda x: attrition[x])
final_data['MaritalStatus'] = final_data['MaritalStatus'].map(lambda x: married[x])


final_data = pd.get_dummies(final_data, columns =['EducationField'])


final_data = final_data.iloc[:,:-1]
final_data = pd.get_dummies(final_data, columns =['JobRole'])
final_data = final_data.iloc[:,:-1]
final_data.isnull().sum()

final_data = final_data.dropna()
final_data = final_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis = 1)
#data2.to_csv("Data.csv", index = False)

data3 = pd.read_csv("../input/hr-analytics/Data_12.csv")
data3.head()
data3 = data3.dropna()
data3.head()
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.style.use("seaborn")

plt.figure(figsize=(15,8))
color = {"Yes": 'red', 'No': 'blue'}
plt.scatter(data3['Age'], data3['MonthlyIncome'], c =data3['Attrition'].apply(lambda x: color[x]), alpha = 0.8 )
plt.title('Age vs Income')
plt.xlabel("Age")
plt.ylabel("Income")
red_patch = mpatches.Patch(color='red', label='Attrition - Yes')
blue_patch = mpatches.Patch(color='blue', label='Attrition - NO')
plt.legend(handles=[red_patch, blue_patch], loc = 'upper right')

plt.show()
data3.isnull().sum()
plt.figure(figsize = (15,20))
attrition_JS = data3[data3['Attrition'] == "Yes"]['TotalWorkingYears'].value_counts()
attrition_NJS = data3[data3['Attrition'] == "No"]['TotalWorkingYears'].value_counts()

dataframe2 = pd.DataFrame({'Attrition':attrition_JS, 'No Attrition':attrition_NJS})

dataframe2.plot(kind='bar', stacked= 'True', figsize = (20,10))
plt.xlabel("Working Years in a Company")
plt.ylabel("Count of people")
plt.xticks(rotation = 0)
plt.show()
df = data3[['Attrition', 'MonthlyIncome','JobRole']].groupby(['JobRole','Attrition']).mean().reset_index()
df.head()
f1 = df['Attrition'] == "Yes"
df1 = df[f1]
f2 = df['Attrition'] == "No"
df2 = df[f2]
df
fig, ax = plt.subplots()

# Example data
people = df2['JobRole']

y_pos = np.arange(len(people))


ax.barh(y_pos, df2['MonthlyIncome'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Monthly Income')
ax.set_title('Income vs Attrition (No)')

plt.show()

fig, ax = plt.subplots()

# Example data
people = df1['JobRole']

y_pos = np.arange(len(people))


ax.barh(y_pos, df1['MonthlyIncome'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Monthly Income')
ax.set_title('Income vs Attrition (Yes)')

plt.show()

plt.figure(figsize = (15,20))
attrition_JS = df[df['Attrition'] == "Yes"]['MonthlyIncome'].value_counts()
attrition_NJS = df[df['Attrition'] == "No"]['MonthlyIncome'].value_counts()

dataframe2 = pd.DataFrame({'No Attrition':attrition_NJS})

dataframe2.plot(kind='barh', stacked= 'True', figsize = (20,10))
plt.xlabel("Working Years in a Company")
plt.ylabel("Count of people")
plt.xticks(rotation = 0)
plt.show()
attrition_JS = data3[data3['Attrition'] == "Yes"]['JobSatisfaction_D'].value_counts()
attrition_NJS = data3[data3['Attrition'] == "No"]['JobSatisfaction_D'].value_counts()

dataframe2 = pd.DataFrame({'Attrition':attrition_JS, 'No Attrition':attrition_NJS})

dataframe2.plot(kind='bar', stacked= 'True')
plt.show()
plt.hist([data3[data3['Attrition']=="Yes"]['Age'],data3[data3['Attrition']=="No"]['Age']], bins = 10, stacked = True,color = ['red', 'blue'], edgecolor = 'black')
plt.title("Age Group")
plt.xlabel("Age")
plt.ylabel("Count")

red_patch = mpatches.Patch(color='red', label='Attrition - Yes')
blue_patch = mpatches.Patch(color='blue', label='Attrition - NO')
plt.legend(handles=[red_patch, blue_patch], loc = 'upper right')

plt.show()
plt.show()
data3.isnull().sum()
plt.figure(figsize = (5,14))
plt.hist([data3[data3['Attrition']=="Yes"]['StockOptionLevel'],data3[data3['Attrition']=="No"]['StockOptionLevel']], bins = 3, stacked = True,color = ['red', 'blue'], edgecolor = 'black')
plt.title("Marital Status Group")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
plt.figure(figsize = (5,14))
plt.hist([data3[data3['Attrition']=="Yes"]['MaritalStatus'],data3[data3['Attrition']=="No"]['MaritalStatus']], bins = 3, stacked = True,color = ['red', 'blue'], edgecolor = 'black')
plt.title("Marital Status Group")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
plt.figure(figsize = (8,10))
plt.hist([data3[data3['Attrition']=="Yes"]['Average_Time'],data3[data3['Attrition']=="No"]['Average_Time']], bins = 10, stacked = True,color = ['red', 'blue'], edgecolor = 'black')
plt.title("Average Working Hours")
plt.xlabel("Average Time")
plt.ylabel("Count")
red_patch = mpatches.Patch(color='red', label='Attrition - Yes')
blue_patch = mpatches.Patch(color='blue', label='Attrition - NO')
plt.legend(handles=[red_patch, blue_patch], loc = 'upper right')

plt.show()
final_data.head()
X = final_data.drop(['Attrition'], axis = 1)
y = final_data.iloc[:,1]
final_data.head()
from sklearn.neighbors import KNeighborsClassifier


KNN_model = KNeighborsClassifier()

param_grid = {'n_neighbors': [2,4,5,6,7,8,9,10,15,20]}
grid = GridSearchCV(KNN_model, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
from sklearn.svm import SVC


svc = SVC()

param_grid = {'kernel': ['linear','rbf'], 'C':[1]}
    
grid = GridSearchCV(svc, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

param_grid = {'n_estimators':[50,100,150], 'max_depth':[10,20,30,40,50,60,70,80,90]}

grid = GridSearchCV(rf, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.3)

rf = RandomForestClassifier(n_estimators = 100, max_depth = 20)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
y_pred = rf.predict(X)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, y_pred)
cm
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize= (10,10), dpi=100)

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')
final_data.head()
%matplotlib inline
import matplotlib.pyplot as plt
pd.crosstab(data3['JobRole'],data3['Attrition']).plot(kind='bar')
plt.title('Turnover Frequency for Job ROle')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')
plt.savefig('department_bar_chart')
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

param_grid = {'n_estimators':[50,100,150], 'learning_rate':[0.05,0.1,0.5,1]}

grid = GridSearchCV(ada, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

param_grid = {}
grid = GridSearchCV(nb, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

param_grid = {'activation': ['relu'], 'hidden_layer_sizes':[1, 10, 15, 25, 50, 100, [2,10], [3,15],[10,15]]}
grid = GridSearchCV(mlp, param_grid, cv = 10,scoring='accuracy',
                    return_train_score=True)
grid.fit(X, y)

print("Best Parameter: {}".format(grid.best_params_))
print("Best Cross Vlidation Score: {}".format(grid.best_score_))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'newton-cg')

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("accuracy score: {}".format(accuracy_score(y_test, y_pred)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'newton-cg', max_iter=2000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("accuracy score: {}".format(accuracy_score(y_test, y_pred)))
