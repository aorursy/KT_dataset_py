import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',37)
df=pd.read_csv('../input/ibm-hr-analytics-classification/IBM_HR.csv')
df.head()
df.isnull().sum()[df.isnull().sum()!=0]
Null_values_percentage=(df.isnull().sum().sum()/len(df))*100
Null_values_percentage
### Inference: As there is only 1.5% of total null values in dataset, we will drop those null values
df=df.dropna()
df.shape
df.drop_duplicates(keep='first',inplace=True)
df.shape
df.info()
df['Age'].value_counts()
sns.distplot(df['Age'],hist=True,kde=True,color='k',bins=10)
# Majority of employees lie between the age range of 30 to 40
sns.catplot(x='Age',hue='Attrition',data=df,kind='count',height=15)
# Majority of attritions can be seen in 28 to 33 age group range
df['Attrition'].value_counts()
sns.countplot(x='Attrition',data=df,hue='Gender')
# Count of male employees are more in case of attrition
df['BusinessTravel'].value_counts()
sns.countplot(x='BusinessTravel',data=df,hue='Attrition')
sns.catplot(x='BusinessTravel',data=df,hue='Attrition',col='Department',kind='count',height=5)
# Wrt all the departments we can conclude that 'Travel_Frequently Business Travel' are in the verge towards attrition for HR Dept.
df['DailyRate'].value_counts()
sns.distplot(df['DailyRate'],bins=10,color='k')
df['DailyRate'].mean()
df['DailyRate'].min()
df['DailyRate'].max()
# The average of daily rate is somewhere around 802,minimum is 102,and maximum is 1499.
df['Department'].value_counts()
sns.countplot(df['Department'])
# Around 60% employees are working in R&D Department
sns.catplot(x='Department',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Sales department has a high attrition rate
df['DistanceFromHome'].value_counts()
# As from info it is observed that 'Distance From Home' is object type,so we converted it to numeric type
df['DistanceFromHome']=pd.to_numeric(df['DistanceFromHome'],errors='coerce')
plt.figure(figsize=(15,10))
plt.xticks(rotation='vertical')
sns.countplot(df['DistanceFromHome'])
sns.distplot(df['DistanceFromHome'],color='k',bins=10)
# From the above count plot we can see that there are multiple instances of some numbers in int and float,so we will convert all to a single datatype
df['DistanceFromHome']=df['DistanceFromHome'].astype('int')
df['DistanceFromHome'].value_counts()
plt.figure(figsize=(15,10))
plt.xticks(rotation='vertical')
sns.countplot(df['DistanceFromHome'])
sns.distplot(df['DistanceFromHome'],color='k',bins=10)
df['DistanceFromHome'].mean()
df['DistanceFromHome'].min()
df['DistanceFromHome'].max()
# We can see that the avg distance from home is around 9Km, minimum is 1Km and maximum is 29Km.
sns.catplot(x='DistanceFromHome',hue='Attrition',col='Gender',data=df,kind='count',height=15,aspect=0.5)
# In case of both male and female,attrition rate tends to be higher when the distance exceed 10Km.
sns.catplot(x='DistanceFromHome',hue='Attrition',col='Department',data=df,kind='count',height=15,aspect=0.5)
# In case of all departments,attrition rate tends to be higher when the distance exceed 10Km.
df['Education'].value_counts()
sns.countplot(df['Education'])
# Around 30% of employees have education level of 3
sns.catplot(x='Education',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# For both male and female,attrition rate is higher for education level 1,2 and 3.
df['EducationField'].value_counts()
# As there is only 1 count in 'Test' category,so we will impute it in 'Other' category.
df.loc[df['EducationField']=='Test','EducationField']='Other'
df['EducationField'].value_counts()
plt.xticks(rotation='vertical')
sns.countplot(df['EducationField'])
# Around 70% of employees are having 'Life Sciences' and 'Medical' education field.
sns.catplot(x='EducationField',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Attrition rate of female in 'HR' education field is less when compared to male,
# Attrition rate of female in 'Life Sciences' and 'Medical' is more when compared to male.
df['EmployeeCount'].value_counts()
df['EmployeeNumber'].value_counts()
df['Application ID'].value_counts()
df['EnvironmentSatisfaction'].value_counts()
sns.countplot(df['EnvironmentSatisfaction'])
# Count of environment satisfaction is more towards 3 and 4.
sns.catplot(x='EnvironmentSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# For both male and female, attrition rate is high environment satisfaction is 1 and 2. 
df['Gender'].value_counts()
sns.countplot(df['Gender'])
# Approximately female and male ratio is 3:2
sns.catplot(x='Gender',hue='Attrition',kind='count',data=df,height=5)
# For better inference, lets calculate male and female attrition rate.
df.loc[(df['Gender']=='Female') & (df['Attrition']=='Voluntary Resignation')]
Female_Attrition_Rate=1420/9283
Female_Attrition_Rate
df.loc[(df['Gender']=='Male') & (df['Attrition']=='Voluntary Resignation')]
Male_Attrition_Rate=2243/13907
Male_Attrition_Rate
# Hence, Male attrition rate is slightly higher than Female attrition rate.
df['HourlyRate'].value_counts()
# From info we can see that HourlyRate has dtype as object, so lets convert it in integer form
df.info()
df['HourlyRate']=df['HourlyRate'].astype('int')
df.info()
sns.distplot(df['HourlyRate'],color='k',bins=10)
df['HourlyRate'].mean()
df['HourlyRate'].min()
df['HourlyRate'].max()
# Avg hourly rate is around 65 and min hourly rate is 65 and max hourly rate is 100
sns.catplot(x='HourlyRate',hue='Attrition',kind='count',data=df,height=15,aspect=1)
# There is no clear evidence that HourlyRate has any impact on attrition of employees.
df['JobInvolvement'].value_counts()
sns.countplot(df['JobInvolvement'])
# Majority of employees lie in the job involvement 2 and 3
sns.catplot(x='JobInvolvement',hue='Attrition',col='Gender',data=df,kind='count')
# Job involvement 3 has slighly more attrition rate than others.
df['JobLevel'].value_counts()
sns.countplot(df['JobLevel'])
# Majority of employees lie in the job level 1 and 2
sns.catplot(x='JobLevel',hue='Attrition',col='Gender',data=df,kind='count')
# Attrition rate is higher in job level 1 and 2.
df['JobRole'].value_counts()
plt.xticks(rotation='vertical')
sns.countplot(df['JobRole'])
# Count of employees is more in job role as Sales Executive,Laboratory Technician,Research Scientist.
g=sns.catplot(x='JobRole',hue='Attrition',col='Gender',data=df,kind='count',height=7)
g.set_xticklabels(rotation=90)
# Job role as Sales Representative has the highest attrition rate for both male and female,
# Job role as HR has high rate of attrition in case of female gender.
df['JobSatisfaction'].value_counts()
sns.countplot(df['JobSatisfaction'])
# Job Satisfaction count for 3 and 4 are more than 1 and 2.
sns.catplot(x='JobSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition rate can be seen in Job Satisfaction level 1 and 2.
df['MaritalStatus'].value_counts()
sns.countplot(df['MaritalStatus'])
# Count of married employees is more
sns.catplot(x='MaritalStatus',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Attrition rate in singles are higher for both male and female
df['MonthlyIncome'].value_counts()
# As,monthly income column has object dtype, we need to convert it in integer form.
df['MonthlyIncome']=df['MonthlyIncome'].astype('int')
sns.distplot(df['MonthlyIncome'],bins=10,color='k')
df['MonthlyIncome'].mean()
df['MonthlyIncome'].min()
df['MonthlyIncome'].max()
# Minimum monthly income of employees is 1009 and maximum monthly income of employees is 19999 and avg monthly income of employees is 6507.
# Majority of employees are having monthly income lower than 5000.
df['MonthlyRate'].value_counts()
sns.distplot(df['MonthlyRate'],20,color='k')
df['MonthlyRate'].mean()
df['MonthlyRate'].min()
df['MonthlyRate'].max()
# Avg monthly rate of employees is around 14302,min monthly rate is 2094 and max monthly rate is 26999.
df['NumCompaniesWorked'].value_counts()
sns.countplot(df['NumCompaniesWorked'])
# Maximum employees have worked in only 1 company.
sns.catplot(x='NumCompaniesWorked',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# It can be observed that employees who have worked in 1 company have higher attrition rate
df['Over18'].value_counts()
df['OverTime'].value_counts()
sns.countplot(df['OverTime'])
# Approximately ratio of employees doing overtime and employees not doing overtime is 30:70
sns.catplot(x='OverTime',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# A very high attrition rate is seen in employees who are doing overtime for both male and female.
sns.catplot(x='OverTime',hue='Gender',data=df,kind='count',height=7)
# Male has a higher attrition rate in both cases
df['PercentSalaryHike'].value_counts()
sns.countplot(df['PercentSalaryHike'])
# Majority of employees got a salary hike less than 15%
sns.catplot(x='PercentSalaryHike',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition is observed in cases where the salary hike is less than 16% for male when compared to female.
df['PerformanceRating'].value_counts()
sns.countplot(df['PerformanceRating'])
# There are very few employees who have performance rating 4.
sns.catplot(x='PerformanceRating',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Performance Rating 3 has higher rate of attrition for both male and female.
df['RelationshipSatisfaction'].value_counts()
sns.countplot(df['RelationshipSatisfaction'])
# Count of employees having relationship satisfaction 3,4 are more than 1,2.
sns.catplot(x='RelationshipSatisfaction',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition is observed in lower relationship satisfaction for both genders
df['StandardHours'].value_counts()
df['StockOptionLevel'].value_counts()
sns.countplot(df['StockOptionLevel'])
# There are many employees who does not have stock options level,
# As the stock options level increases the count of employees reduces.
sns.catplot(x='StockOptionLevel',hue='Attrition',col='Gender',data=df,kind='count',height=7)
# Higher attrition rate is observed in lower stock options level for both genders.
df['TotalWorkingYears'].value_counts()
sns.distplot(df['TotalWorkingYears'],bins=10,color='k')
plt.figure(figsize=(10,10))
plt.xticks(rotation='vertical')
sns.countplot(df['TotalWorkingYears'])
# Maximum number of employees have total working years as 10 and the count decreases gradually after 10 years.
sns.catplot(x='TotalWorkingYears',hue='Attrition',data=df,kind='count',height=15)
# Higher attrition rate is observed for employees having total working years less than 10 years.
df['TrainingTimesLastYear'].value_counts()
sns.countplot(df['TrainingTimesLastYear'])
# Maximum employees where trained 2 to 3 times since last year
sns.catplot(x='TrainingTimesLastYear',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Higher attrition rate can be seen where number of trainings given to employees are less for both gender.
df['WorkLifeBalance'].value_counts()
sns.countplot(df['WorkLifeBalance'])
# Count of employees having worklife balance as 3 is more wrt others
sns.catplot(x='WorkLifeBalance',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# Lower work life balance has somewhat high rate of attrition
sns.catplot(x='WorkLifeBalance',hue='Attrition',col='Department',data=df,kind='count',height=7)
# HR Department has less attrition rate in any cases of work life balance
df['YearsAtCompany'].value_counts()
sns.distplot(df['YearsAtCompany'],bins=20,color='k')
# Count of employees is maximum who have worked less than 8 years
sns.catplot(x='YearsAtCompany',hue='Attrition',data=df,kind='count',height=15)
# We can see higher attrition rate for those employees who have worked for less than 10 years
df['YearsInCurrentRole'].value_counts()
sns.distplot(df['YearsInCurrentRole'],bins=20,color='k')
# Count of employees having 2 to 3 years in current role are more.
sns.catplot(x='YearsInCurrentRole',hue='Attrition',data=df,kind='count',height=10)
# After 5 years in same role,attrition rate gradually decreases with increase in years.
df.info()
df['YearsSinceLastPromotion'].value_counts()
sns.distplot(df['YearsSinceLastPromotion'],bins=20,color='k')
sns.countplot(df['YearsSinceLastPromotion'])
# Majority of employees are in the category of having 0,1 or 2 years since last promotion.
sns.catplot(x='YearsSinceLastPromotion',hue='Attrition',data=df,kind='count',height=10)
# Attrition rate is higher where Years since last promotion is less than 7
df['YearsWithCurrManager'].value_counts()
sns.distplot(df['YearsWithCurrManager'],bins=20,color='k')
plt.figure(figsize=(10,7))
plt.xticks(rotation='vertical')
sns.countplot(df['YearsWithCurrManager'])
# Majority of employees areworking with their manager for around 2 years.
sns.catplot(x='YearsWithCurrManager',hue='Attrition',data=df,kind='count',height=10)
# As the employees work for more years with same manager,they get mentally attached with that manager and have a good comfort zone.
# Hence, they get retained for a longer period of time.
# But there are a few exceptions where the attrition rate is high even if the years are more.This maybe due to internal disputes.So,regular counselling should be done.
df['Employee Source'].value_counts()
# Since there is only 1 entry in Test,we will simply shift in other group
df.loc[df['Employee Source']=='Test','Employee Source']='Company Website'
df['Employee Source'].value_counts()
plt.xticks(rotation='vertical')
sns.countplot(df['Employee Source'])
# Around 25% employee source is Company Website, so we should management to emhance its worth more.
sns.catplot(x='Employee Source',hue='Attrition',col='Gender',data=df,kind='count',height=10)
# At the same time,it is observed that the maximum attrition is taking place for those employees who have joined organization through companies website.
# Hence, reality check should be done in the website.
df.head()
df.shape
df.info()
df1=df.drop(['EmployeeCount','EmployeeNumber','Application ID','StandardHours','Over18'],axis=1)
df1.shape
df1.head()
df1['Attrition']=df1['Attrition'].apply(lambda x:1 if x=='Voluntary Resignation' else 0)
plt.figure(figsize=(20,15))
ax = sns.heatmap(df1.corr(),cmap='rainbow',mask=abs(df1.corr())<0.05,annot=True)
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
# Impact of Age on Attrition of employees
sns.catplot(x='Age',hue='Attrition',data=df,kind='count',height=15)
# Impact of Job Level on Attrition of employees
sns.catplot(x='JobLevel',hue='Attrition',data=df,kind='count')
# Impact of Marital Status on Attrition of employees
sns.catplot(x='MaritalStatus',hue='Attrition',data=df,kind='count',height=7)
# Monthly Income affecting Attrition rate:
sns.barplot(x='Attrition',y='MonthlyIncome',data=df)
sns.relplot(x='JobInvolvement',y='MonthlyIncome',hue='Attrition',data=df,size='MonthlyIncome')
# Business Travel affecting attrition rate
sns.countplot(x='BusinessTravel',hue='Attrition',data=df)
df1.head()
# For chosing outliers we will only chose continous feature
# Lets check the value counts of all the features
for i in df1.columns:
    print(i)
    print('value_counts :-','\n',df[i].value_counts(),'\n'*3)
list=['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','TotalWorkingYears','YearsAtCompany',
      'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
for i in list:
    sns.boxplot(y=df1[i])
    plt.show()
# We will use Z-score to remove outliers
import scipy.stats as st
outliers = st.zscore(df1[['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','TotalWorkingYears'
                          ,'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']])
df1 = df1[(abs(outliers)<3).all(axis=1)]
df1.head()
for i in list:
    sns.boxplot(y=df1[i])
    plt.show()
for i in list:
    print(i,' : ',df[i].skew())
for i in list:
    sns.distplot(df[i])
    plt.show()
# We will do boxcox transformation for fixing the skewness of the dataset
for i in list:
    df1[i]=st.boxcox(df1[i]+1)[0]
df1.skew()
df2=df1.copy()
df2.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df2['BusinessTravel'].value_counts()
# We will do get dummies or ohe for this column
df2['Gender'].value_counts()
df2['Gender']=le.fit_transform(df2['Gender'])
df2['JobRole'].value_counts()
# We will do get dummies or ohe for this column
df2['JobSatisfaction'].value_counts()
df2['JobSatisfaction']=df2['JobSatisfaction'].astype('int')
df2.info()
df2['MaritalStatus'].value_counts()
# We will do get dummies or ohe for this column
df2['OverTime'].value_counts()
df2['OverTime']=le.fit_transform(df2['OverTime'])
df2['PercentSalaryHike'].value_counts()
df2['PercentSalaryHike']=df2['PercentSalaryHike'].astype('int')
df2['Employee Source'].value_counts()
# We will do get dummies or ohe for this column
df2.info()
df2=pd.get_dummies(df2,drop_first=True)
df2.head()
Unscaled_data=df2.drop('Attrition',axis=1)
Unscaled_data
plt.figure(figsize=(50,35))
ax = sns.heatmap(df2.corr(),annot=True,mask=abs(df2.corr())<0.05)
bottom,top = ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
# Now our dataset is cleaned and ready for processing
X=df2.drop('Attrition',axis=1)
y=df2['Attrition']
import statsmodels.api as sm
X_con=sm.add_constant(X)
model=sm.Logit(y,X_con).fit()
result=model.summary()
result
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from tpot import TPOTClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

# Splitting dataset in train and test:
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=0)
# Data Scaling using standard scaler
# Apply classifier
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('lr',LogisticRegression())])
pipeline_dt=Pipeline([('scaler2',StandardScaler()),
                     ('dt',DecisionTreeClassifier())])
pipeline_rf=Pipeline([('scalar3',StandardScaler()),
                     ('rfc',RandomForestClassifier())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('knn',KNN())])
pipeline_xgbc=Pipeline([('scalar5',StandardScaler()),
                     ('xgboost',XGBClassifier())])
pipeline_lgbc=Pipeline([('scalar6',StandardScaler()),
                     ('lgbc',lgb.LGBMClassifier())])
pipeline_ada=Pipeline([('scalar7',StandardScaler()),
                     ('adaboost',AdaBoostClassifier())])
pipeline_sgdc=Pipeline([('scalar8',StandardScaler()),
                     ('sgradient',SGDClassifier())])
pipeline_nb=Pipeline([('scalar9',StandardScaler()),
                     ('nb',GaussianNB())])
pipeline_extratree=Pipeline([('scalar10',StandardScaler()),
                     ('extratree',ExtraTreesClassifier())])
pipeline_svc=Pipeline([('scalar11',StandardScaler()),
                     ('svc',SVC())])
pipeline_gbc=Pipeline([('scalar12',StandardScaler()),
                     ('GBC',GradientBoostingClassifier())])
# Lets make the list of pipelines
pipelines=[pipeline_lr,pipeline_dt,pipeline_rf,pipeline_knn,pipeline_xgbc,pipeline_lgbc,pipeline_ada,
           pipeline_sgdc,pipeline_nb,pipeline_extratree,pipeline_svc,pipeline_gbc]
pipe_dict={0:'Logistic Regression',1:'Decision Tree',2:'RandomForestClassifier',3:'KNN',4:'XGBC',5:'LGBC',6:'ADA',7:'SGDC',8:'NB',9:'ExtraTree',10:'SVC',11:'GBC'}
# Let's check whether the target variable is balanced or not:
sns.countplot(df2['Attrition'])
df2['Attrition'].value_counts()
# As the dataset is highly imbalanced, we will use SMOTE:
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_sample(X_train,y_train)
print(X_sm.shape, y_sm.shape)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report,roc_auc_score, roc_curve
lr=LogisticRegression()
lr.fit(X_sm,y_sm)
y_train_pred=lr.predict(X_sm)
y_train_prob=lr.predict_proba(X_sm)[:,1]

y_test_pred=lr.predict(X_test)
y_test_prob=lr.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_sm,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_sm,y_train_pred))
print('Classification Report-Train\n',classification_report(y_sm,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_sm,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,threshold= roc_curve(y_test,y_test_prob)
threshold[0]=threshold[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax2=ax1.twinx()
ax2.plot(fpr,threshold,'-g')
ax2.set_ylabel('TRESHOLD')
plt.show()
plt.show()
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_train_pred=lr.predict(X_train)
y_train_prob=lr.predict_proba(X_train)[:,1]

y_test_pred=lr.predict(X_test)
y_test_prob=lr.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,threshold= roc_curve(y_test,y_test_prob)
threshold[0]=threshold[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax2=ax1.twinx()
ax2.plot(fpr,threshold,'-g')
ax2.set_ylabel('TRESHOLD')
plt.show()
plt.show()
# train models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# logistic regression
model1 = LogisticRegression()

# knn
model2 = KNeighborsClassifier()

# Random Forest Classifier
model3 = RandomForestClassifier()

# XGBClassifier
model4=XGBClassifier()

# fit model
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(X_test)
pred_prob2 = model2.predict_proba(X_test)
pred_prob3 = model3.predict_proba(X_test)
pred_prob4 = model4.predict_proba(X_test)
from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])

print(auc_score1, auc_score2,auc_score3, auc_score4)
# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='Random Forest')
plt.plot(fpr4, tpr4, linestyle='--',color='black', label='XGBC')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
for i in pipelines:
    i.fit(X_sm,y_sm)
    y_pred=i.predict(X_test)
    print('Classification Report : ', i[1] ,'\n',(classification_report(y_test,y_pred)))
    print('f1-score : ', i[1],' : ',(f1_score(y_test,y_pred)))
    print('\n'*2,'------------------------------------------------------------------------------------------------')
knn=KNN(n_neighbors=9)
knn.fit(X_train,y_train)
y_train_pred=knn.predict(X_train)
y_train_prob=knn.predict_proba(X_train)[:,1]

y_test_pred=knn.predict(X_test)
y_test_prob=knn.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TRP')
ax2=ax1.twinx()
ax2.plot(fpr,thresholds,'-g')
ax2.set_ylabel('TRESHOLDS')
plt.show()
plt.show()
rf=RandomForestClassifier(max_depth=15, min_samples_leaf=10, min_samples_split=20,
                       n_estimators=5)
rf.fit(X_train,y_train)
y_train_pred=rf.predict(X_train)
y_train_prob=rf.predict_proba(X_train)[:,1]

y_test_pred=rf.predict(X_test)
y_test_prob=rf.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax2=ax1.twinx()
ax2.plot(fpr,thresholds,'-g')
ax2.set_ylabel('THRESHOLD')
plt.show()
plt.show()
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
y_train_pred=xgbc.predict(X_train)
y_train_prob=xgbc.predict_proba(X_train)[:,1]

y_test_pred=xgbc.predict(X_test)
y_test_prob=xgbc.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TRP')
ax2=ax1.twinx()
ax2.plot(fpr,thresholds,'-g')
ax2.set_ylabel('TRESHOLDS')
plt.show()
plt.show()
pipeline=[DecisionTreeClassifier(),RandomForestClassifier(),XGBClassifier(),
        ExtraTreesClassifier()]
for i in pipeline:
    i.fit(X,y)
    i.feature_importances_
    print(i)
    imp_features = pd.Series(i.feature_importances_,index=X.columns)
    plt.figure(figsize =(10,10))
    imp_features.nlargest(8).sort_values(ascending=True).plot(kind='barh')

    plt.show()
a=[]
for i in pipeline:
    i.fit(X,y)
    i.feature_importances_
    imp_features = pd.Series(i.feature_importances_,index=X.columns)
    x = pd.DataFrame(imp_features.nlargest(8).sort_values(ascending=False))
    a.append(x.index.values)
    b=pd.DataFrame(a)
c=b.T
c
c[0]
d=pd.DataFrame()
for i in c.columns:
    d=pd.concat([d,c[i]],ignore_index=True)
print(d)
d = d.rename(columns={0: 'Imp_Features'})
d
d['Imp_Features'].value_counts()
d['Imp_Features'].unique()
df2
new_X=df2[['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome',
       'TrainingTimesLastYear', 'TotalWorkingYears', 'MonthlyRate',
       'HourlyRate', 'PercentSalaryHike','BusinessTravel_Travel_Frequently', 'OverTime', 'StockOptionLevel']]
new_X
new_y=df2['Attrition']
new_y
X
X_train,X_test,y_train,y_test=train_test_split(new_X,new_y, test_size=0.3, random_state=0)
rf=RandomForestClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=20,
                       n_estimators=5)
rf.fit(X_train,y_train)
y_train_pred=rf.predict(X_train)
y_train_prob=rf.predict_proba(X_train)[:,1]

y_test_pred=rf.predict(X_test)
y_test_prob=rf.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,threshold= roc_curve(y_test,y_test_prob)
threshold[0]=threshold[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax2=ax1.twinx()
ax2.plot(fpr,threshold,'-g')
ax2.set_ylabel('THRESHOLD')
plt.show()
plt.show()
from sklearn.model_selection import GridSearchCV
rfgrid=GridSearchCV(estimator=RandomForestClassifier(),
                   param_grid=[{'n_estimators': [5,10],
                               'max_depth':[5,10,15],
                               'min_samples_leaf':[10,50,100],
                               'min_samples_split': [20,100,200]}])
rfgrid_fit=rfgrid.fit(X_train,y_train)
print(rfgrid_fit.best_estimator_)
rfgrid_score=rfgrid_fit.score(X_train,y_train)
rfgrid_score
pred=rfgrid_fit.predict(X_test)
pred
rfgrid_score_test=rfgrid_fit.score(X_test,pred)
rfgrid_score_test
y_train_pred=rfgrid.predict(X_train)
y_train_prob=rfgrid.predict_proba(X_train)[:,1]

y_test_pred=rfgrid.predict(X_test)
y_test_prob=rfgrid.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,threshold= roc_curve(y_test,y_test_prob)
threshold[0]=threshold[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax2=ax1.twinx()
ax2.plot(fpr,threshold,'-g')
ax2.set_ylabel('THRESHOLD')
plt.show()
plt.show()
from sklearn.model_selection import RandomizedSearchCV
rfrandomized=RandomizedSearchCV(estimator=RandomForestClassifier(),
                   param_distributions=[{'n_estimators': [1,5,10],
                               'max_depth':[5, 10,15],
                               'min_samples_leaf':[5,10, 50, 100],
                               'min_samples_split': [10,20,100,200]}])
rfrand_fit=rfrandomized.fit(X_train,y_train)
print(rfrand_fit.best_estimator_)
rfrand_score=rfrand_fit.score(X_train,y_train)
rfrand_score
y_train_pred=rfrand_fit.predict(X_train)
y_train_prob=rfrand_fit.predict_proba(X_train)[:,1]

y_test_pred=rfrand_fit.predict(X_test)
y_test_prob=rfrand_fit.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,threshold= roc_curve(y_test,y_test_prob)
threshold[0]=threshold[0]-1
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax2=ax1.twinx()
ax2.plot(fpr,threshold,'-g')
ax2.set_ylabel('THRESHOLD')
plt.show()
plt.show()
dfpca=df2.drop('Attrition',axis=1)
dfpca
from sklearn.decomposition import PCA
# Create scaler: scaler
scaler = StandardScaler()
scaler.fit(dfpca)
# transform
data_scaled = scaler.transform(dfpca)
data_scaled
pca = PCA()

# fit PCA
pca.fit(data_scaled)
pd.DataFrame({'Eigenvalue':pca.explained_variance_,'Proporsion Explained':pca.explained_variance_ratio_,'Cummumlative Proportion Exaplained':np.cumsum(pca.explained_variance_ratio_)})

plt.figure(figsize=(15,8))
plt.bar(range(1,53),pca.explained_variance_ratio_)
plt.plot(range(1,53),np.cumsum(pca.explained_variance_ratio_),'r')
plt.show()
# PCA features
features = range(pca.n_components_)
features
# PCA transformed data
data_pca = pca.transform(data_scaled)
data_pca.shape
# PCA components variance ratios.
pca.explained_variance_ratio_
plt.figure(figsize=(15,12))
plt.bar(features, pca.explained_variance_ratio_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
pca2 = PCA(n_components=20, svd_solver='full')

# fit PCA
pca2.fit(data_scaled)

# PCA transformed data
data_pca2 = pca2.transform(data_scaled)
data_pca2.shape
Xpca=pd.DataFrame(data_pca2)
ypca=df2['Attrition']
Xpca.shape,ypca.shape
X_train,X_test,y_train,y_test=train_test_split(Xpca,ypca, test_size=0.3, random_state=0)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_train_pred=rf.predict(X_train)
y_train_prob=rf.predict_proba(X_train)[:,1]

y_test_pred=rf.predict(X_test)
y_test_prob=rf.predict_proba(X_test)[:,1]

print('Confusion Matrix-Train\n',confusion_matrix(y_train,y_train_pred))
print('Accuracy Score-Train\n',accuracy_score(y_train,y_train_pred))
print('Classification Report-Train\n',classification_report(y_train,y_train_pred))
print('AUC Score-Train\n',roc_auc_score(y_train,y_train_prob))
print('\n'*2)
print('Confusion Matrix-Test\n',confusion_matrix(y_test,y_test_pred))
print('Accuracy Score-Test\n',accuracy_score(y_test,y_test_pred))
print('Classification Report-Test\n',classification_report(y_test,y_test_pred))
print('AUC Score-Test\n',roc_auc_score(y_test,y_test_prob))
print('\n'*3)
print('Plot : AUC-ROC Curve')
fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
fig,ax1 = plt.subplots()
ax1.plot(fpr,tpr)
ax1.plot(fpr,fpr)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TRP')
ax2=ax1.twinx()
ax2.plot(fpr,thresholds,'-g')
ax2.set_ylabel('TRESHOLDS')
plt.show()
plt.show()
