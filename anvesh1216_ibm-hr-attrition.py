# Importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
#Reading the Data Set
hr_data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Displaying the Top 5 rows in Data Set
hr_data.head()
#Check if any Column's has NA or Missing Values
hr_data.isnull().any()
#Shape of the Data
hr_data.shape
#Data Types of Columns
hr_data.dtypes
#Show how much % of employees left the organization
hr_data.Attrition.value_counts(normalize=True)
plt.figure(figsize=(8,6))
Attrition=hr_data.Attrition.value_counts()
sb.barplot(x=Attrition.index ,y=Attrition.values)
plt.title('Distribution of Employee Turnover')
plt.xlabel('Employee Turnover', fontsize=16)
plt.ylabel('Count', fontsize=16)
sb.distplot(hr_data['Age'])
sb.distplot(hr_data['MonthlyIncome'])
sb.distplot(hr_data['TotalWorkingYears'])
#Method that plot density plots on the columns passed as input
def kdePlot(var):
    fig = plt.figure(figsize=(15,4))
    ax=sb.kdeplot(hr_data.loc[(hr_data['Attrition'] == 'No'),var] , color='b',shade=True, label='no Attrition') 
    ax=sb.kdeplot(hr_data.loc[(hr_data['Attrition'] == 'Yes'),var] , color='r',shade=True, label='Attrition')
    plt.title('Employee Attrition with respect to {}'.format(var))
    
numerical_df=hr_data.select_dtypes(include=np.number)
numeric_cols_kdeplot=list(numerical_df.columns)
remove_columns=['Age','DistanceFromHome','Education','EmployeeCount','EmployeeNumber',
'EnvironmentSatisfaction' ,'HourlyRate','JobInvolvement','JobSatisfaction','MonthlyRate','NumCompaniesWorked',
'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
'TrainingTimesLastYear','WorkLifeBalance','YearsSinceLastPromotion']
for l in remove_columns:
    numeric_cols_kdeplot.remove(l)
#Plotting KDE plots
for n in numeric_cols_kdeplot:
    kdePlot(n)
BarPlot_columns=['Age','DistanceFromHome','EducationField',
                'JobInvolvement','JobLevel','JobRole','OverTime','TotalWorkingYears','TrainingTimesLastYear',
                'WorkLifeBalance','YearsInCurrentRole']
#Method the perform Bar plots
def Bar_plots(var):
    col=pd.crosstab(hr_data[var],hr_data.Attrition)
    col.div(col.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False, figsize=(8,4))
    plt.xticks(rotation=90)
for col in BarPlot_columns:
    Bar_plots(col)
#Replacing Yes with 1 and No with 0 in Attrition Column
hr_data['Attrition']=np.where(hr_data['Attrition']=='No', #condition
                 0, #value if condition is true
                 1)
hr_data.describe().iloc[:,:20]
hr_data.EmployeeCount.value_counts()
hr_data.StandardHours.value_counts()
len(set(hr_data.EmployeeNumber))
hr_data=hr_data.drop(['EmployeeCount', 'StandardHours','EmployeeNumber'], axis=1)
hr_data.groupby('Attrition').mean().iloc[:,:20]
hr_data.groupby('Attrition').mean().iloc[:,20:26]
corr_matrix = hr_data.corr()
f , ax = plt.subplots(figsize=(20,12))
sb.heatmap(corr_matrix,vmax=0.8, annot=True)
numerical_df=hr_data.select_dtypes(include=np.number)
categorical_df=hr_data.select_dtypes(exclude=np.number)
numeric_cols=list(numerical_df.columns)
categorical_cols=list(categorical_df.columns)
for n in categorical_cols:
    print(pd.crosstab(hr_data['Attrition'],hr_data[n],normalize='columns'))
categorical_df_dummies=pd.get_dummies(hr_data[categorical_cols],drop_first=True)
final_df=pd.concat([categorical_df_dummies,numerical_df],axis=1)
final_df.head()
y=final_df.Attrition
X=final_df.drop(['Attrition'], axis=1)
#Splitting Data in Train and Test Set
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,precision_recall_curve,confusion_matrix,precision_score,confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
#Building base model(Predicting that no employee leaves the company)
base=np.zeros(1470)
print(accuracy_score(base,hr_data.Attrition))
#Method that applies model on the data and Predict the attrition
def model(mod,model_name,x_tr,y_tr,x_tes,y_te):
    mod.fit(x_tr,y_tr)
    pred_dt=mod.predict(x_tes)
    print("     ",model_name,"      ")
    print("Accuracy ",accuracy_score(pred_dt,y_te))
    print("ROC_AUC  ",roc_auc_score(pred_dt,y_te))
    cm=confusion_matrix(pred_dt,y_te)
    print("Confusion Matrix  \n",cm)
    print("                    Classification Report \n",classification_report(pred_dt,y_te))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model(lr,"Logistic Regression",X_train,y_train,X_test,y_test)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(min_samples_leaf=20, max_depth=4)
model(dt,"Decision Tree",X_train,y_train,X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,max_depth=4)
model(rf,"Random Forest",X_train,y_train,X_test,y_test)
#Performing OverSample using SMOTE(Synthetic Minority Over Sampling Technique)
from imblearn.over_sampling import SMOTE
smote=SMOTE()
X_sm, y_sm=smote.fit_sample(X,y)
X_train_sm,X_test_sm,y_train_sm,y_test_sm=train_test_split(X_sm,y_sm,test_size=0.2,random_state=100)
#logistic Regression for OverSampled Data 
lr_sm=LogisticRegression()
model(lr_sm,"Logistic Regression",X_train_sm,y_train_sm,X_test_sm,y_test_sm)
dt_sm=DecisionTreeClassifier(min_samples_leaf=20, max_depth=4)
model(dt_sm,"Decision Tree",X_train_sm,y_train_sm,X_test_sm,y_test_sm)
rf_sm=RandomForestClassifier(n_estimators=10,max_depth=4)
model(rf_sm,"Random Forest",X_train_sm,y_train_sm,X_test_sm,y_test_sm)



