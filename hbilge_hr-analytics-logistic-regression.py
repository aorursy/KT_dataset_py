import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from datetime import timedelta
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
employee_survey_data = pd.read_csv('../input/hr-analytics-case-study/employee_survey_data.csv')
general_data = pd.read_csv('../input/hr-analytics-case-study/general_data.csv')
start_time = pd.read_csv('../input/hr-analytics-case-study/in_time.csv')
manager_survey_data = pd.read_csv('../input/hr-analytics-case-study/manager_survey_data.csv')
finish_time = pd.read_csv('../input/hr-analytics-case-study/out_time.csv')
print(general_data.head())
print(employee_survey_data.head())
print(manager_survey_data.head())
print(start_time.head())
print(finish_time.head())
print('General data shape:', general_data.shape)
print('Employee survey data shape:', employee_survey_data.shape)
print('Manager survey data shape:', manager_survey_data.shape)
print('Start working time data shape', start_time.shape)
print('End working time data shape:', finish_time.shape)
# Firstly, change column name Unnamed: 0 to EmployeeID in start and end time datasets.
start_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
finish_time.rename(columns={'Unnamed: 0':'EmployeeID'}, inplace=True)
print('Number of unique values of EmployeeID in start time dataset:', start_time.EmployeeID.nunique())
print('Number of unique values of EmployeeID in finish time dataset:', finish_time.EmployeeID.nunique())
general_data.set_index('EmployeeID', inplace=True)
employee_survey_data.set_index('EmployeeID', inplace=True)
manager_survey_data.set_index('EmployeeID', inplace=True)
start_time.set_index('EmployeeID', inplace=True)
finish_time.set_index('EmployeeID', inplace=True)
main_data = pd.concat([general_data, employee_survey_data, manager_survey_data], axis = 1)
print(main_data.columns.values)
start_time = start_time.apply(pd.to_datetime)
finish_time = finish_time.apply(pd.to_datetime)
main_data['WorkingHours'] = (finish_time - start_time).mean(axis=1)
main_data['WorkingHours'] = main_data['WorkingHours'] / np.timedelta64(1, 's')
main_data['Overtime'] = main_data['WorkingHours'] - main_data['StandardHours'] * 3600
print(main_data.info())
print('\033[1mNULL VALUES\033[0m\n'+ str(main_data.isnull().sum()))
plt.figure(figsize=(25,8))

plt.subplot(1,5,1)
main_data['NumCompaniesWorked'].plot(kind='density', color='teal')
plt.title('Density Plot Of Number Of \nCompanies Worked')

plt.subplot(1,5,2)
main_data['TotalWorkingYears'].plot(kind='density', color='blue')
plt.title('Density Plot Of \nTotal Working Years')

plt.subplot(1,5,3)
main_data['EnvironmentSatisfaction'].plot(kind='density', color='teal')
plt.title('Density Plot Of \nEnvironment Satisfaction')

plt.subplot(1,5,4)
main_data['JobSatisfaction'].plot(kind='density', color='blue')
plt.title('Density Plot Of \nJob Satisfaction')

plt.subplot(1,5,5)
main_data['WorkLifeBalance'].plot(kind='density', color='green')
plt.title('Density Plot Of \nWork Life Balance')

plt.show()
null = ['NumCompaniesWorked', 'TotalWorkingYears', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
for i in null:
    main_data[i] = main_data[i].fillna(main_data[i].median())
print('\033[1mNULL VALUES\033[0m\n'+ str(main_data.isnull().values.any()))
for i in main_data:
    print("data[\'" + i + "\']:", main_data[i].nunique())
main_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)
total = len(main_data['Attrition'])
Attrition = pd.DataFrame(main_data['Attrition'].value_counts())
print(Attrition.T)
plt.figure(figsize=(6,4))
plt.style.use('ggplot')
ax = Attrition.plot(kind='bar', color='pink')
plt.title('Attrition', fontweight='bold', fontsize=15)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 10 , '{:.0%}'.format(height/total))
plt.show()
main_data['AgeGroups'] = pd.cut(main_data['Age'], range(10, 70, 10))
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
sns.distplot(main_data['Age'], color='green')
plt.xlim(10,70)
plt.title('Age Distribution')

plt.subplot(1,3,2)
main_data['MaritalStatus'].value_counts().plot(kind='bar', color='lightblue')
plt.xticks(rotation=0)
plt.title('Marital Status Distribution')

plt.subplot(1,3,3)
main_data['Gender'].value_counts().plot(kind='bar', color='lightpink')
plt.xticks(rotation=0)
plt.title('Gender Distribution')

plt.show()
plt.figure(figsize=(16,10))

plt.subplot(2,3,4)
main_data['Department'].value_counts().plot(kind='bar', color='lightblue')
plt.xticks(rotation=0)
plt.title('Department Distribution')

plt.subplot(2,3,5)
main_data['JobRole'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Job Role Distribution')

plt.subplot(2,3,6)
main_data['EducationField'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Education Field Distribution')

plt.show()
graphs = ['AgeGroups', 'MaritalStatus', 'Gender', 'Department', 'JobRole', 'EducationField']
plt.figure(figsize=(20,15))
for index, item in enumerate(graphs):
    plt.subplot(2,3,index+1)
    ax = sns.countplot(x=item, hue='Attrition', data=main_data, palette='husl')
    if index+1>3: plt.xticks(rotation=90)
    index = int(len(ax.patches)/2)
    for left,right in zip(ax.patches[:index], ax.patches[index:]):
        left_height = left.get_height()
        right_height = right.get_height()
        total = left_height + right_height
        ax.text(left.get_x() + left.get_width()/2., left_height + 20, '{:.0%}'.format(left_height/total), ha="center")
        ax.text(right.get_x() + right.get_width()/2., right_height + 20, '{:.0%}'.format(right_height/total), ha="center")
plt.show()  
main_data = main_data.drop('AgeGroups', axis=1)
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(main_data[main_data['Attrition']=='Yes']['MonthlyIncome'], color='darkblue')
plt.title('Distribution of Monthly Income for Attrition (YES)', fontsize=12, fontweight='bold')

plt.subplot(1,2,2)
sns.distplot(main_data[main_data['Attrition']=='No']['MonthlyIncome'], color='darkred')
plt.title('Distribution of Monthly Income for Attrition (NO)', fontsize=12, fontweight='bold')

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(main_data[main_data['Attrition']=='Yes']['PercentSalaryHike'], color='darkgreen')
plt.title('Distribution of Percent Salary Hike for Attrition (YES)', fontsize=12, fontweight='bold')

plt.subplot(1,2,2)
sns.distplot(main_data[main_data['Attrition']=='No']['PercentSalaryHike'], color='darkorange')
plt.title('Distribution of Percent Salary Hike for Attrition (NO)', fontsize=12, fontweight='bold')

plt.show()
plt.figure(figsize=(16,6))

sns.kdeplot(main_data['YearsSinceLastPromotion'][main_data.Attrition=='Yes'], color='blue', shade=True)
sns.kdeplot(main_data['YearsSinceLastPromotion'][main_data.Attrition=='No'], color='red', shade=True)
plt.title('Distribution of Years Since Last Promotion', fontsize=15)
plt.legend(['Attrition (YES)', 'Attrition (NO)'])
plt.xlabel('Years', fontsize=12)

plt.show()
plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.kdeplot(main_data['YearsAtCompany'][main_data.Attrition=='Yes'], shade=True, color='green')
sns.kdeplot(main_data['YearsAtCompany'][main_data.Attrition=='No'], shade=True, color='red')
plt.title('Distribution Of Years At Company', fontsize=13)
plt.ylabel('Distribution')
plt.legend(['Attrition (YES)','Attrition (NO)'])

plt.subplot(1,2,2)
sns.kdeplot(main_data['YearsWithCurrManager'][main_data.Attrition=='Yes'], shade=True, color='green')
sns.kdeplot(main_data['YearsWithCurrManager'][main_data.Attrition=='No'], shade=True, color='red')
plt.title('Distribution of Years With Current Manager', fontsize=13)
plt.legend(['Attrition (YES)','Attrition (NO)'])


plt.show()
plt.figure(figsize=(16,6))

plt.subplot(1,3,1)
sns.violinplot(data=main_data, x='Attrition', y='JobSatisfaction', palette='Blues')
plt.title('Job Satisfaction')

plt.subplot(1,3,2)
sns.violinplot(data=main_data, x='Attrition', y='EnvironmentSatisfaction', palette='Blues')
plt.title('Environment Satisfaction')

plt.subplot(1,3,3)
sns.violinplot(data=main_data, x='Attrition', y='WorkLifeBalance', palette='Blues')
plt.title('Work Life Balance')

plt.show()
numerical_columns = main_data.select_dtypes(exclude='object').columns
numerical_data = main_data[numerical_columns]
categorical_columns = main_data.select_dtypes(include='object').columns
categorical_data = main_data[categorical_columns]
plt.figure(figsize=(20,15))
for index, item in enumerate(numerical_columns, 1):
    plt.subplot(5, 4, index)
    sns.boxplot(main_data[item])
plt.show() 
columns = ['MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
           'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'PerformanceRating', 'WorkingHours']
main_data[columns] = (main_data[columns] + 1).transform(np.log)
dummies = pd.get_dummies(main_data[categorical_columns], drop_first = True)
main_data = pd.concat([main_data, dummies], axis = 1)
main_data.drop(categorical_columns, axis = 1, inplace = True)
X = main_data.drop('Attrition_Yes', axis=1)
y = main_data['Attrition_Yes']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
columns = X.columns
main_data[columns] = scaler.fit_transform(main_data[columns])
plt.rcParams['figure.figsize'] = [35,30]
sns.heatmap(main_data.corr(), cmap='PuBu', annot=True, linewidths=.5, annot_kws={'size':8})
plt.title('Correlation Matrix', fontweight='bold', fontsize='15')
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["variables"] = X.columns
vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
for index,column in enumerate(X.columns):
    print(index, column, vif['vif'][index])
    if vif['vif'][index]>5:
        vif = vif.drop([index], axis=0)
print(vif)
columns = list(vif['variables'])
data = main_data[columns]
data = pd.concat([data, main_data['Attrition_Yes']], axis=1)
from sklearn.model_selection import train_test_split
X = data.drop('Attrition_Yes', axis=1)
y = data ['Attrition_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
lr = LogisticRegression(solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
lrpred = lr.predict(X_test)
print('Accuracy score of Logistic Regression:' + str(accuracy_score(y_test,lrpred)))
print('Confusion Matrix\n' + str(confusion_matrix(y_test, lrpred)))
plt.figure(figsize=(12,5))

lrprob = lr.predict_proba(X_test)
lr_pred = lrprob[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, lr_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.title('Logistic Regression ROC', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.xlabel('False Positive Rate', fontsize=15)
plt.legend(loc = 'lower right', prop={'size': 14})

plt.show()
from sklearn.metrics import classification_report
print('Logistic Regression\n',classification_report(y_test, lrpred))