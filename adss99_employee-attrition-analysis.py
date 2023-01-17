import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

sns.set(style="ticks", color_codes=True)

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
emp_gen_info = pd.read_csv("../input/hr-analytics-case-study/general_data.csv")

emp_survey = pd.read_csv("../input/hr-analytics-case-study/employee_survey_data.csv")

man_survey = pd.read_csv("../input/hr-analytics-case-study/manager_survey_data.csv")

in_time = pd.read_csv("../input/hr-analytics-case-study/in_time.csv")

out_time = pd.read_csv("../input/hr-analytics-case-study/out_time.csv")
print("emp_gen_info:",emp_gen_info.shape)

print("emp_survey:",emp_survey.shape)

print("man_survey:",man_survey.shape)
emp_gen_info.head()
emp_survey.head()
man_survey.head()
emp_gen_info.set_index('EmployeeID', inplace=True)

emp_survey.set_index('EmployeeID', inplace=True)

man_survey.set_index('EmployeeID', inplace=True)
Employee =  pd.concat([emp_gen_info, emp_survey, man_survey], axis = 1)

Employee.head()
Employee.T.apply(lambda columns: columns.nunique(), axis=1)
Employee.drop(['EmployeeCount', 'Over18','StandardHours'], axis=1,inplace = True)
Employee.dtypes
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

Employee['BusinessTravel'] = labelEncoder_X.fit_transform(Employee['BusinessTravel'])

Employee['Department'] = labelEncoder_X.fit_transform(Employee['Department'])

Employee['EducationField'] = labelEncoder_X.fit_transform(Employee['EducationField'])

Employee['Gender'] = labelEncoder_X.fit_transform(Employee['Gender'])

Employee['JobRole'] = labelEncoder_X.fit_transform(Employee['JobRole'])

Employee['MaritalStatus'] = labelEncoder_X.fit_transform(Employee['MaritalStatus'])

Employee['Attrition'] = labelEncoder_X.fit_transform(Employee['Attrition'])
Employee.dtypes
Employee.isnull().any()
meanOfNumCompaniesWorked = Employee["NumCompaniesWorked"].astype('float').mean(axis = 0 )

Employee["NumCompaniesWorked"].replace(np.nan,meanOfNumCompaniesWorked,inplace = True)



meanOfTotalWorkingYears = Employee["TotalWorkingYears"].astype('float').mean(axis = 0 )

Employee["TotalWorkingYears"].replace(np.nan,meanOfTotalWorkingYears,inplace = True)



meanOfEnvironmentSatisfaction = round(Employee["EnvironmentSatisfaction"].astype('float').mean(axis = 0 ))

Employee["EnvironmentSatisfaction"].replace(np.nan,meanOfEnvironmentSatisfaction,inplace = True)



meanOfJobSatisfaction = round(Employee["JobSatisfaction"].astype('float').mean(axis = 0 ))

Employee["JobSatisfaction"].replace(np.nan,meanOfJobSatisfaction,inplace = True)



meanOfWorkLifeBalance = round(Employee["WorkLifeBalance"].astype('float').mean(axis = 0 ))

Employee["WorkLifeBalance"].replace(np.nan,meanOfWorkLifeBalance,inplace = True)

Employee.head()
import datetime as dt
print("in_time:",in_time.shape)

in_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)

in_time.set_index('EmployeeID', inplace=True)

in_time.head()
print("out_time:",in_time.shape)

out_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)

out_time.set_index('EmployeeID', inplace=True)

out_time.head()
in_time_stamp = in_time.apply(pd.to_datetime) #converting into timestamp object

out_time_stamp = out_time.apply(pd.to_datetime)

df_working_hours = (out_time_stamp - in_time_stamp)# time spent in the company

df_working_hours.head()

df_working_time= df_working_hours / np.timedelta64(1, 'h') #converting time spent to float value

df_working_time.head()

mean_working_hours = df_working_time.mean(axis=1)# mean time spend in company in above period

Employee['MeanWorkingHours'] = mean_working_hours

emp_gen_info['MeanWorkingHours'] = mean_working_hours #** adding to the dataframe
import calendar
in_time_stamp.head()

week_day = in_time_stamp.dropna(how='all', axis=1) # drop columns with every value as NaT as they are public holidays

week_day.head()

col_list =list( week_day.columns)#converting columns to list

df_col_list = pd.DataFrame(col_list)#coverting list to pandas dataframe

df_col_list_stamp = df_col_list.apply(pd.to_datetime)#convert df to datetime object

df_col_list_stamp[0] = df_col_list_stamp[0].dt.weekday # df to day of week 0 for monday and 6 for sunday

weekday_list=list(df_col_list_stamp[0])# creating list of weekdays for renaming columns

week_day.columns = [weekday_list]#renaming columns

Employee['TotalLeave']=week_day.isna().sum(axis=1)#Adding total leaves to the employee dataframe

emp_gen_info['TotalLeave']=week_day.isna().sum(axis=1)

Employee['LeaveMonFri']=week_day[[0,4]].isna().sum(axis=1) # leave on monday and friday

emp_gen_info['LeaveMonFri']=week_day[[0,4]].isna().sum(axis=1)
Employee.hist(figsize=(30,20),grid = False);
plt.figure(figsize=(25,6))



plt.subplot(1,3,1)

sns.countplot(x='BusinessTravel', hue='Attrition', data=emp_gen_info, palette='pastel');

plt.title('BusinessTravel')



plt.subplot(1,3,2)

sns.countplot(x='Department', hue='Attrition', data=emp_gen_info, palette='pastel');

plt.title('Department')



plt.subplot(1,3,3)

sns.countplot(x='StockOptionLevel', hue='Attrition', data=emp_gen_info, palette='pastel');

plt.title('StockOptionLevel')



plt.show()

plt.figure(figsize=(25,6))



plt.subplot(1,3,1)

sns.kdeplot(emp_gen_info['Age'][emp_gen_info.Attrition=='Yes'], shade=True, color='orangered')

sns.kdeplot(emp_gen_info['Age'][emp_gen_info.Attrition=='No'], shade=True, color='royalblue')

plt.title('Distribution Of Age', fontsize=13)

plt.ylabel('Probability Density')

plt.legend(['Attrition (YES)','Attrition (NO)'])



plt.subplot(1,3,2)

sns.kdeplot(emp_gen_info['DistanceFromHome'][emp_gen_info.Attrition=='Yes'], shade=True, color='orangered')

sns.kdeplot(emp_gen_info['DistanceFromHome'][emp_gen_info.Attrition=='No'], shade=True, color='royalblue')

plt.title('Distribution of Distance From Home', fontsize=13)

plt.legend(['Attrition (YES)','Attrition (NO)'])



plt.subplot(1,3,3)

sns.kdeplot(emp_gen_info['TotalWorkingYears'][emp_gen_info.Attrition=='Yes'], shade=True, color='orangered')

sns.kdeplot(emp_gen_info['TotalWorkingYears'][emp_gen_info.Attrition=='No'], shade=True, color='royalblue')

plt.title('Distribution of Total Working Years', fontsize=13)

plt.legend(['Attrition (YES)','Attrition (NO)'])





plt.show()
plt.figure(figsize=(25,6))



plt.subplot(1,3,1)

sns.violinplot(data=Employee, x='Attrition', y='JobSatisfaction', palette='pastel')

plt.title('Job Satisfaction')



plt.subplot(1,3,2)

sns.violinplot(data=Employee, x='Attrition', y='EnvironmentSatisfaction', palette='pastel')

plt.title('Environment Satisfaction')



plt.subplot(1,3,3)

sns.violinplot(data=Employee, x='Attrition', y='JobInvolvement', palette='pastel')

plt.title('JobInvolvement')



plt.show()


matrix = np.triu(Employee.corr())

plt.figure(figsize = (25, 15))

sns.heatmap(Employee.corr(), annot = True, linewidth = 0.02,cmap = 'RdYlGn', mask=matrix)

plt.show()
plt.figure(figsize=(24,8))

plt.subplot(1,3,1)

sns.boxplot(x='Attrition', y='MeanWorkingHours', data=Employee, palette='pastel');

plt.subplot(1,3,2)

sns.boxplot(x='Attrition', y='Age', data=Employee, palette='pastel');

plt.subplot(1,3,3)

sns.boxplot(x='Attrition', y='TotalWorkingYears', data=Employee, palette='pastel');

plt.show()

plt.figure(figsize=(25,6))

sns.kdeplot(emp_gen_info['MeanWorkingHours'][emp_gen_info.Attrition=='Yes'], shade=True, color='orangered');

sns.kdeplot(emp_gen_info['MeanWorkingHours'][emp_gen_info.Attrition=='No'], shade=True, color='royalblue');

plt.title('Distribution Of MeanWorkingHours', fontsize=13)

plt.ylabel('Probability Density')

plt.legend(['Attrition (YES)','Attrition (NO)']);

from sklearn.preprocessing import StandardScaler, Normalizer

scaler = StandardScaler()

scaler.fit(Employee)

Employee_scaled = scaler.transform(Employee)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

Employee_pca = pca.fit_transform(Employee_scaled)

Employee_pca.shape
plt.figure(figsize=(10,8))



plt.scatter(Employee_pca[:,0],Employee_pca[:,1],c=Employee['Attrition']);

from sklearn.manifold import TSNE, Isomap

iso = Isomap(n_components=3, n_neighbors=20)

Employee_iso = iso.fit_transform(Employee_scaled)

Employee_scaled.shape
plt.figure(figsize=(10,8))

plt.scatter(Employee_iso[:,0],Employee_iso[:,1],c=Employee['Attrition']);
from sklearn.model_selection import train_test_split
y = Employee['Attrition']

x = Employee.drop('Attrition', axis = 1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
X_train_lr,X_test_lr, y_train_lr, y_test_lr = train_test_split(x,y, test_size = 0.20, random_state=39)
X_train_lr = scaler.fit_transform(X_train_lr)

X_test_lr = scaler.fit_transform(X_test_lr)

print(X_train_lr.shape,X_test_lr.shape)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_lr,y_train_lr)

LR
yhat_LR = LR.predict(X_test_lr)

print(yhat_LR[0:5])

print(y_test_lr[0:5])
from sklearn import metrics

from sklearn.metrics import f1_score

from sklearn.metrics import jaccard_similarity_score
print("LogisticRegression's Accuracy: ", metrics.accuracy_score(y_test_lr,yhat_LR))

print("FI SCORE: ", f1_score(y_test_lr, yhat_LR, average='weighted') )

print("jaccard_similarity_score: ", jaccard_similarity_score(y_test_lr, yhat_LR)) 
from sklearn.tree import DecisionTreeClassifier
X_train_dt,X_test_dt, y_train_dt, y_test_dt= train_test_split(x,y, test_size = 0.20, random_state=41)
X_train_dt = scaler.fit_transform(X_train_dt)

X_test_dt = scaler.fit_transform(X_test_dt)

print(X_train_dt.shape,X_test_dt.shape)
DT = DecisionTreeClassifier(criterion="entropy", max_depth = 10)

DT.fit(X_train_dt,y_train_dt)
yhat_DT = DT.predict(X_test_dt)

print (yhat_DT [0:5])

print (y_test_dt [0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test_dt,yhat_DT))

print("FI SCORE: ", f1_score(y_test_dt, yhat_DT, average='weighted') )

print("jaccard_similarity_score: ", jaccard_similarity_score(y_test_dt, yhat_DT)) 
from sklearn.ensemble import RandomForestClassifier
X_train_rf,X_test_rf, y_train_rf, y_test_rf= train_test_split(x,y, test_size = 0.20, random_state=41)
X_train_rf = scaler.fit_transform(X_train_rf)

X_test_rf = scaler.fit_transform(X_test_rf)

print(X_train_rf.shape,X_test_rf.shape)
RF = RandomForestClassifier()

RF.fit(X_train_rf,y_train_rf)
yhat_RF = RF.predict(X_test_rf)

print (yhat_RF [0:5])

print (y_test_rf [0:5])
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_test_rf,yhat_RF))

print("FI SCORE: ", f1_score(y_test_rf, yhat_RF, average='weighted') )

print("jaccard_similarity_score: ", jaccard_similarity_score(y_test_rf, yhat_RF)) 
importances = RF.feature_importances_

std = np.std([tree.feature_importances_ for tree in RF.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1][:40]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train_rf.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

fig = plt.figure(figsize=(25, 10));

plt.title("Relative Feature importances")

plt.bar(range(X_train_rf.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train_rf.shape[1]), indices)

plt.xlim([-1, X_train_rf.shape[1]])

plt.show();
plt.figure(figsize=(25,6))

plt.subplot(1,3,1)

sns.boxplot(x='Attrition', y='MeanWorkingHours', data=Employee, palette='pastel');

plt.subplot(1,3,2)

sns.boxplot(x='Attrition', y='Age', data=Employee, palette='pastel');

plt.subplot(1,3,3)

sns.boxplot(x='Attrition', y='TotalWorkingYears', data=Employee, palette='pastel');

plt.show()
plt.figure(figsize=(25,6))

plt.subplot(1,3,1)

sns.boxplot(x='Attrition', y='MonthlyIncome', data=Employee, palette='pastel');

plt.subplot(1,3,2)

sns.boxplot(x='Attrition', y='YearsAtCompany', data=Employee, palette='pastel');

plt.subplot(1,3,3)

sns.boxplot(x='Attrition', y='DistanceFromHome', data=Employee, palette='pastel');

plt.show()