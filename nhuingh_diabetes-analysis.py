# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('max_rows',800)

df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.info()
#EDA

pregnancies=df.groupby('Outcome').agg({'Pregnancies':['describe']})

pregnancies.index=['Non-Diabetic','Diabetic']



pregnancies
#EDA

plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='Pregnancies',data=df,palette='muted')

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,6)

plt.xlabel('Diabetic or not')

plt.ylabel('No. of Pregnancies')

plt.title("Pregnancies")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
glucose=df.groupby('Outcome').agg({'Glucose':['describe']})

glucose.index=['Non-Diabetic','Diabetic']



glucose
plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='Glucose',data=df,palette='Set1')

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,160)

plt.xlabel('Diabetic or not')

plt.ylabel('Glucose Level')

plt.title("Gluose Level")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
blood_pressure=df.groupby('Outcome').agg({'BloodPressure':['describe']})

blood_pressure.index=['Non-Diabetic','Diabetic']



blood_pressure
plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='BloodPressure',data=df,palette='colorblind')

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,80)

plt.xlabel('Diabetic or not')

plt.ylabel('Blood Pressure Level')

plt.title("Blood Pressure Level")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
Skin_Thickness=df.groupby('Outcome').agg({'SkinThickness':['describe']})

Skin_Thickness.index=['Non-Diabetic','Diabetic']



Skin_Thickness
plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='SkinThickness',data=df,palette='hls')

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,26)

plt.xlabel('Diabetic or not')

plt.ylabel('Skin Thickness')

plt.title("Skin Thickness")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
Insulin=df.groupby('Outcome').agg({'Insulin':['describe']})

Insulin.index=['Non-Diabetic','Diabetic']



Insulin
plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='Insulin',data=df)

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,130)

plt.xlabel('Diabetic or not')

plt.ylabel('Insulin Level')

plt.title("Insulin Level")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
BMI=df.groupby('Outcome').agg({'BMI':['describe']})

BMI.index=['Non-Diabetic','Diabetic']



BMI
plt.figure(figsize=(8,8))

sns.barplot(x='Outcome',y='BMI',data=df)

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,40)

plt.xlabel('Diabetic or not')

plt.ylabel('BMI value')

plt.title("Body Mass Index(BMI)")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
Age=df.groupby('Outcome').agg({'Age':['describe']})

Age.index=['Non-Diabetic','Diabetic']



Age
df1=df.copy()

df1['Age']=pd.cut(df1['Age'],bins=5)

df1['Outcome'].replace({1:'Diabetic',0:'Non-Diabetic'},inplace=True)



plt.figure(figsize=(10,10))

plt.subplot(2,1,1)

sns.countplot(x='Age',hue='Outcome',data=df1)

plt.xticks([0,1,2,3,4],['Young-Adults','Middle-Aged','Retirement-Age','Old','Very Old'])

plt.xlabel('Age Groups')

plt.ylabel('NUmber of Diabetic and Non-Diabetic')

plt.title("Age Groups")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)





plt.subplot(2,1,2)

sns.barplot(x='Outcome',y='Age',data=df)

plt.xticks([0,1],['Non-Diabetic','Diabetic'])

plt.ylim(0,40)

plt.xlabel('Diabetic or not')

plt.ylabel('Age')

plt.title("Age")

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.tight_layout()
filt1=(df['Outcome']==0)&(df['Glucose']==0)

filt2=(df['Outcome']==1)&(df['Glucose']==0)

df.loc[filt1,'Glucose']=110

df.loc[filt2,'Glucose']=141



filt1=(df['Outcome']==0)&(df['BloodPressure']==0)

filt2=(df['Outcome']==1)&(df['BloodPressure']==0)

df.loc[filt1,'BloodPressure']=68

df.loc[filt2,'BloodPressure']=71



filt1=(df['Outcome']==0)&(df['SkinThickness']==0)

filt2=(df['Outcome']==1)&(df['SkinThickness']==0)

df.loc[filt1,'SkinThickness']=20

df.loc[filt2,'SkinThickness']= 22



filt1=(df['Outcome']==0)&(df['Insulin']==0)

filt2=(df['Outcome']==1)&(df['Insulin']==0)

df.loc[filt1,'Insulin']=69

df.loc[filt2,'Insulin']=100



filt1=(df['Outcome']==0)&(df['BMI']==0)

filt2=(df['Outcome']==1)&(df['BMI']==0)

df.loc[filt1,'BMI']=30

df.loc[filt2,'BMI']=37
from sklearn.preprocessing import MinMaxScaler

df_modified=df.copy()

df_modified.drop('Outcome',axis=1,inplace=True)

cols=df_modified.columns



scaler=MinMaxScaler()



df_modified=pd.DataFrame(scaler.fit_transform(df_modified))

df_modified.columns=cols

df_modified['Outcome']=df['Outcome']

df_modified['Outcome'].replace({0:'Non-Diabetic',1:'Diabetic'},inplace=True)
x=sns.PairGrid(df_modified,hue='Outcome',vars=cols,despine=True,palette='colorblind',layout_pad=True)

x.map_offdiag(sns.scatterplot)

x.add_legend()

x
plt.figure(figsize=(15,15))

sns.heatmap(data=df_modified[cols].corr(),xticklabels=True,yticklabels=True,cbar=True,linecolor='white',annot=True)
#Model Building

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score



xtrain,xtest,ytrain,ytest=train_test_split(df[cols],df['Outcome'],test_size=0.33,random_state=0)


#RandomForest



from sklearn.ensemble import RandomForestClassifier



clf1=RandomForestClassifier(random_state=0,n_estimators=100,max_depth=100,max_features='auto')

clf1.fit(xtrain,ytrain)

predictions1=clf1.predict(xtest)

probabilities=clf1.predict_proba(xtest)

importance1=list(zip(xtrain.columns,clf1.feature_importances_))



print('accuracy'+' '+'='+' '+str(accuracy_score(ytest,predictions1)))

print('ROC score'+' '+'='+' '+str(roc_auc_score(ytest,probabilities[:,1])))

print('f1 score'+' '+'='+' '+str(f1_score(ytest,predictions1)))

print('recall score'+' '+'='+' '+str(recall_score(ytest,predictions1)))

print('precision score'+' '+'='+' '+str(precision_score(ytest,predictions1)))

print()

print()

print("Feature Importances:")

print(importance1)
#Gradient boosted Classifier



from sklearn.ensemble import GradientBoostingClassifier



clf2= GradientBoostingClassifier(random_state=0,learning_rate=0.07,n_estimators=100,max_depth=3,max_features='auto')

clf2.fit(xtrain,ytrain)

predictions2=clf2.predict(xtest)

probabilities1=clf2.predict_proba(xtest)

importance2=list(zip(xtrain.columns,clf2.feature_importances_))



print('accuracy'+' '+'='+' '+str(accuracy_score(ytest,predictions2)))

print('ROC score'+' '+'='+' '+str(roc_auc_score(ytest,probabilities1[:,1])))

print('f1 score'+' '+'='+' '+str(f1_score(ytest,predictions2)))

print('recall score'+' '+'='+' '+str(recall_score(ytest,predictions2)))

print('precision score'+' '+'='+' '+str(precision_score(ytest,predictions2)))

print()

print()

print("Feature Importances:")

print(importance2)
from sklearn.svm import SVC



scaler.fit_transform(xtrain)

scaler.transform(xtest)



clf3=SVC(random_state=0,probability=True,C=80.0,gamma=0.00002,kernel='rbf')

clf3.fit(xtrain,ytrain)

predictions3=clf3.predict(xtest)

probabilities2=clf3.predict_proba(xtest)



print('accuracy'+' '+'='+' '+str(accuracy_score(ytest,predictions3)))

print('ROC score'+' '+'='+' '+str(roc_auc_score(ytest,probabilities2[:,1])))

print('f1 score'+' '+'='+' '+str(f1_score(ytest,predictions3)))

print('recall score'+' '+'='+' '+str(recall_score(ytest,predictions3)))

print('precision score'+' '+'='+' '+str(precision_score(ytest,predictions3)))
from sklearn.linear_model import LogisticRegression



scaler.fit_transform(xtrain)

scaler.transform(xtest)



clf4=LogisticRegression(random_state=0,C=0.09,max_iter=10000)

clf4.fit(xtrain,ytrain)

predictions4=clf4.predict(xtest)

probabilities3=clf4.predict_proba(xtest)



print('accuracy'+' '+'='+' '+str(accuracy_score(ytest,predictions4)))

print('ROC score'+' '+'='+' '+str(roc_auc_score(ytest,probabilities3[:,1])))

print('f1 score'+' '+'='+' '+str(f1_score(ytest,predictions4)))

print('recall score'+' '+'='+' '+str(recall_score(ytest,predictions4)))

print('precision score'+' '+'='+' '+str(precision_score(ytest,predictions4)))
from sklearn.metrics import roc_curve



fpr,tpr,thresholds=roc_curve(ytest,probabilities[:,1])

fpr1,tpr1,thresholds1=roc_curve(ytest,probabilities1[:,1])

fpr2,tpr2,thresholds2=roc_curve(ytest,probabilities2[:,1])

fpr3,tpr3,thresholds3=roc_curve(ytest,probabilities3[:,1])





plt.figure(figsize=(8,8))

sns.lineplot(y=tpr,x=fpr,ci=None)

sns.lineplot(y=tpr1,x=fpr1,ci=None)

sns.lineplot(y=tpr2,x=fpr2,ci=None,color='black')

sns.lineplot(y=tpr3,x=fpr3,ci=None,color='red')



plt.legend(labels=['Random Forest','Gradient Boosting Classifier','Support Vector Classifier','Logistic Regression'])

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

plt.tight_layout()