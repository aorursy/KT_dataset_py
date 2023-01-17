import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
df=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.dtypes
df.isnull().any()
front=df['Attrition']
df.drop(labels=['Attrition'],axis=1,inplace=True)
df.insert(0,'Attrition',front)
df.head()
#Delete Unwanted Records
df.drop(labels=['EmployeeCount','EmployeeNumber','StockOptionLevel','StandardHours'],axis=1,inplace=True)
df.head()
#df['Gender']=df['Gender'].map({'Male':0,'Female':1}) Map doesnt work
Attrition={'Yes':1,'No':0}
df.Attrition=[Attrition[item] for item in df.Attrition]
#Get categorical values of column 
df.EducationField.unique()
# creating a dict file 
Gender={'Male':1,'Female':0}
# traversing through dataframe Gender column and writing values where key matches
df.Gender=[Gender[item] for item in df.Gender]

Field={'Life Sciences':2,'Medical':1,'Other':0,'Marketing':3,'Technical Degree':4,'Human Resources':5}
df.EducationField=[Field[item] for item in df.EducationField]
#Summary based on Attrition
df1=df.groupby('Attrition')
df1.mean()
corr=df.corr()
corr=(corr)
plt.figure(figsize=(10, 10))
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values,cmap='Blues')
corr

Attrition_Rate=df.Attrition.value_counts()/len(df)
Attrition_Rate
sns.barplot(x='Attrition',y='MonthlyIncome',hue='Gender',data=df,color='green').set_title('Employee Income Gender Distribution')
plt.figure(figsize=(10, 10))
plt.show()
sns.barplot(x='Attrition',y='DistanceFromHome',hue='Gender',data=df,color='blue').set_title('Employee Distance Gender Distribution')
plt.show()
df['Income_Range']=pd.cut(df['MonthlyIncome'],[1000,5000,10000,15000,20000])
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y='Income_Range',hue='Attrition',data=df).set_title('Employee Salary Attrition Distribution')
plt.plot()
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'JobSatisfaction'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'JobSatisfaction'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee Job Satisfaction Rating',ylabel='Frequency')
plt.title('Employee Job Satisfaction Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'WorkLifeBalance'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'WorkLifeBalance'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee WorkLifeBalance Rating',ylabel='Frequency')
plt.title('Employee WorkLifeBalance Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'RelationshipSatisfaction'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'RelationshipSatisfaction'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee RelationshipSatisfaction Rating',ylabel='Frequency')
plt.title('Employee Relationship Satisfaction Rating - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Attrition']==0),'YearsAtCompany'],color='g',shade=False,label='No Attrition')
ax=sns.kdeplot(df.loc[(df['Attrition']==1),'YearsAtCompany'],color='r',shade=True,label='Attrition')
ax.set(xlabel='Employee YearsAtCompany ',ylabel='Frequency')
plt.title('Employee YearsAtCompany - Attrition vs No Attrition')
fig=plt.figure(figsize=(15,8))
value=df['YearsAtCompany']<11
df3=df[value]
sns.countplot(x='YearsAtCompany',hue='Attrition',data=df3)
plt.show()
fig=plt.figure(figsize=(10,6))
sns.countplot(x='YearsWithCurrManager',hue='Attrition',data=df,color='black')
plt.show()
fig=plt.figure(figsize=(10,6))
sns.countplot(x='YearsSinceLastPromotion',hue='Attrition',data=df,color='green')
plt.show()
total_records= len(df)
columns = ["Gender","MaritalStatus","WorkLifeBalance","EnvironmentSatisfaction","JobSatisfaction",
           "JobLevel",'NumCompaniesWorked',"JobInvolvement","BusinessTravel",'Department']

j=0
for i in columns:
    j +=1
    plt.subplot(5,2,j)
    ax1 = sns.countplot(data=df,x= i,hue="Attrition")
    if(j==9 or j== 10):
        plt.xticks( rotation=90)
    for p in ax1.patches:
        height = p.get_height()
        #ax1.text(p.get_x()+p.get_width()/2.,
               # height + 3,
                #'{:1.2f}'.format(height/total_records,0),
                #ha="center",rotation=0) 

# Custom the subplot layout
plt.subplots_adjust(bottom=0.1, top=4)
plt.show()
#Selecting numeric paremeters for Feature Engineering
df3=df[['JobLevel','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance','Attrition']]
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

# Create train and test splits
target_name = 'Attrition'
X = df3.drop('Attrition', axis=1)

y=df3[target_name]
X_train,X_test,y_train,t_test=train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train,y_train)

## plot the importances ##
importances = dtree.feature_importances_
feat_names = df3.drop(['Attrition'],axis=1).columns

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

# Create an intercept term for the logistic regression equation
df['value'] = 1
indep_var = ['JobLevel','JobInvolvement','EnvironmentSatisfaction','value', 'Attrition']
df = df[indep_var]

# Create train and test splits
target_name = 'Attrition'
X = df.drop('Attrition', axis=1)

y=df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)

X_train.head()
import statsmodels.api as sm
iv = ['JobLevel','JobInvolvement','EnvironmentSatisfaction', 'value']
logReg = sm.Logit(y_train, X_train[iv])
answer = logReg.fit()

answer.summary
answer.params
# Create function to compute coefficients
coef = answer.params
def y (coef,JobLevel,JobInvolvement , EnvironmentSatisfaction) : 
    return coef[3] + coef[0]*JobLevel + coef[1]*JobInvolvement + coef[2]*EnvironmentSatisfaction

import numpy as np

# An Employee Having at level 1 and rating 1 for EnvironmentSatisfaction and 1 for JobInvolvement a 54% chance of attrition
y1 = y(coef, 1, 1, 1)
p = np.exp(y1) / (1+np.exp(y1))
p
# Compare the Logistic Regression Model V.S. Decision Tree Model V.S. Random Forest Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from numpy.core.umath_tests import inner1d


#Logistic Regression Model
model1 = LogisticRegression(class_weight="balanced",)
model1.fit(X_train, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, model1.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, model1.predict(X_test)))

#Decision Tree Model
model2=DecisionTreeClassifier(min_weight_fraction_leaf=0.01,class_weight="balanced",)
model2.fit(X_train,y_train)
print("\n\n ---Decision Tree Model ---")
dtree_roc_auc=roc_auc_score(y_test,model2.predict(X_test))
print("Decision Tree AUC = %2.2f" % dtree_roc_auc)
print(classification_report(y_test,model2.predict(X_test)))

#Random Forest Model
model3=RandomForestClassifier( n_estimators=1000,max_depth=None,min_samples_split=10,class_weight="balanced")
model3.fit(X_train,y_train)
print("\n\n --- Random Forest Model ----")
rforest_roc_auc=roc_auc_score(y_test,model2.predict(X_test))
print("Random forest AUC = %2.2f" % rforest_roc_auc)
print(classification_report(y_test,model3.predict(X_test)))


# Using 10 fold Cross-Validation to train Logistic Regression Model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression(class_weight = "balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# Using 10 fold Cross-Validation to train Decision Tree Model
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = DecisionTreeClassifier(class_weight = "balanced",min_weight_fraction_leaf=0.01)
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# Using 10 fold Cross-Validation to train Random Forest Model
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = RandomForestClassifier(n_estimators=1000,max_depth=None,min_samples_split=10,class_weight="balanced")
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model1.predict_proba(X_test)[:,1])
#The first column is the probability that the entry has the -1 label 
#and the second column is the probability that the entry has the +1 label.
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, model3.predict_proba(X_test)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, model2.predict_proba(X_test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rforest_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dtree_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


