import pandas as pd
import numpy as np
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#### Read Dataset
OriginalData=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv');
OriginalData.head(6)
OriginalData.columns.values
OriginalData.shape
OriginalData.info(0)
OriginalData.describe()
#### Pandas Profiling
'''Generates profile reports from a pandas DataFrame. The pandas df.describe() function is great but a little basic for serious exploratory data analysis.<br>
pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.

For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:

Essentials: type, unique values, missing values
Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
Most frequent values
Histogram
Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
Missing values matrix, count, heatmap and dendrogram of missing values
'''
#!pip install pandas-profiling
#import pandas_profiling
#OriginalData.profile_report()
OriginalData.isnull().values.any()

OriginalData.isna().any()
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
sns.countplot(OriginalData['Attrition'], palette="Set2",saturation=10)
numericalData=OriginalData.copy().drop(columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime','Attrition','Over18'])
numericalData.shape
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
numericalData.hist(bins=30,ax=ax);
OriginalData['Attrition'].describe()
dataset=OriginalData.copy()
attritionStatus={'No':0,'Yes':1}
dataset['Attrition']=dataset['Attrition'].map(attritionStatus)
pd.DataFrame(dataset['Attrition'].value_counts()).T
print('Employee not leaving the company : ',round((1233/1470)*100,2),'%')
print('Employee leaving the company : ',round((237/1470)*100,2),'%')
# Data looks clean with no potential outliers.
#We can drop Employee Count and StanardHours features since they are constant and does not contribute to the model.
corrData=dataset.copy().drop(columns=['StandardHours','EmployeeCount'])
corrData.shape
#Using Pearson Correlation
plt.figure(figsize=(20,20))
cor = corrData.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
# Employee number will be used for display purpose
empNo=dataset['EmployeeNumber']
#Target/Respone Variable
response=dataset['Attrition']
# Some variable whose value is not changing, So standard deviation of that variable is Zero. So It is not Significant for analysis.
# Those variable are Employee count, Over18, StandardHours.
dataset=dataset.drop(columns=['EmployeeNumber','Attrition','StandardHours','EmployeeCount','Over18'])
dataset=pd.get_dummies(dataset,columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],drop_first=True)
dataset.columns
dataset.columns.size
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 34 best features
k=35
select_feature = SelectKBest(score_func=chi2,k=k)
select_feature.fit(dataset,response)

dfscores=pd.DataFrame(select_feature.scores_)
dfcolumns=pd.DataFrame(dataset.columns)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns=['Features','Score']

#print 15 best features
print(featureScores.nlargest(k,'Score'))
dataset.columns[select_feature.get_support()]
impFeatures=dataset[['Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
       'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
       'MonthlyRate', 'NumCompaniesWorked', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Research & Development', 'Department_Sales',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Technical Degree', 'JobRole_Human Resources',
       'JobRole_Laboratory Technician', 'JobRole_Manager',
       'JobRole_Manufacturing Director', 'JobRole_Research Director',
       'JobRole_Sales Representative', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'OverTime_Yes']]
impFeatures.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(impFeatures,response,test_size=0.3,random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import StandardScaler
'''
The main idea is to normalize/standardize (mean = 0 and standard deviation = 1) your features before applying machine learning techniques.
StandardScaler performs the task of Standardization. Usually a dataset contains variables that are different in scale. 
For e.g. an Employee dataset will contain AGE column with values on scale 20-70 and SALARY column with values on scale 10000-80000.
As these two columns are different in scale, they are Standardized to have common scale while building machine learning model.
'''
sc_X=StandardScaler()

#Standard scalar removes columns values and indexs after normalization so we have to provide columns values and indexes again.
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_test2=pd.DataFrame(sc_X.transform(X_test))

X_train2.columns=X_train.columns.values
X_test2.columns=X_test.columns.values

X_train2.index=X_train.index.values
X_test2.index=X_test.index.values

X_train=X_train2
X_test=X_test2
from sklearn.linear_model import LogisticRegression

model_Reg=LogisticRegression()
model_Reg.fit(X_train,y_train)
X_test.shape
y_pred_reg = model_Reg.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc=accuracy_score(y_test,y_pred_reg)
prec=precision_score(y_test,y_pred_reg)
rec=recall_score(y_test,y_pred_reg)
f1=f1_score(y_test,y_pred_reg)
confusion_matrix(y_test,y_pred_reg)
# Making the Confusion Matrix
rf_cm = confusion_matrix(y_test,y_pred_reg)

# building a graph to show the confusion matrix results
rf_cm_plot = pd.DataFrame(rf_cm, index = [i for i in {"Attrition", "No Attrition"}],
                  columns = [i for i in {"No attrition", "Attrition"}])
plt.figure(figsize = (6,5))
sns.heatmap(rf_cm_plot, annot=True, vmin=5, vmax=90.5, cbar=False, fmt='g')
results=pd.DataFrame([['Logistic Regression',acc,prec,rec,f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall','F1 Score'])
results