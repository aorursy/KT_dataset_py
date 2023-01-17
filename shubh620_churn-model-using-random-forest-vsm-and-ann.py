

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix , classification_report

from sklearn.metrics import roc_curve , auc

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load Data

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
df.SeniorCitizen.unique()
#Convert to Categorical variable

df.SeniorCitizen= df.SeniorCitizen.apply(lambda x : 'No' if x == 0 else 'Yes')

#Check Type after conversion

df.SeniorCitizen.unique()
df['TotalCharges_new']= pd.to_numeric(df.TotalCharges,errors='coerce')
#Check NULL values after the conversion

df.loc[pd.isna(df.TotalCharges_new),'TotalCharges']
#Fill 11 Missing values from the original column

TotalCharges_Missing=[488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]

df.loc[pd.isnull(df.TotalCharges_new),'TotalCharges_new']=TotalCharges_Missing

#We are good to replace old columns with the new numerical column

df.TotalCharges=df.TotalCharges_new

df.drop(['customerID','TotalCharges_new'],axis=1,inplace=True)

df.info()
df.dtypes=='object'

categorical_var=[i for i in df.columns if df[i].dtypes=='object']

for z in categorical_var:

    print(df[z].name,':',df[z].unique())
Dual_features= ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for i in Dual_features:

    df[i]=df[i].apply(lambda x: 'No' if x=='No internet service' else x)

#Remove No Phones Service that equivilent to No for MultipleLines

df.MultipleLines=df.MultipleLines.apply(lambda x: 'No' if x=='No phone service' else x)

#Check levels or all Categorical Variables

for z in [i for i in df.columns if df[i].dtypes=='object']:

    print(df[z].name,':',df[z].unique())
continues_var=[i for i in df.columns if df[i].dtypes !='object']

fig , ax = plt.subplots(1,3,figsize=(15,5))

for i , x in enumerate(continues_var):

    ax[i].hist(df[x][df.Churn=='No'],label='Churn=0',bins=30)

    ax[i].hist(df[x][df.Churn=='Yes'],label='Churn=1',bins=30)

    ax[i].set(xlabel=x,ylabel='count')

    ax[i].legend()
fig , ax = plt.subplots(1,3,figsize=(15,5))

for i , xi in enumerate(continues_var):

    sns.boxplot(x=df.Churn,y=df[xi],ax=ax[i],hue=df.gender)

    ax[i].set(xlabel='Churn',ylabel=xi)

    ax[i].legend()

#Remove Churn Variable for Analysis

categorical_var_NoChurn= categorical_var[:-1]
#Count Plot all Categorical Variables with Hue Churn

fig , ax = plt.subplots(4,4,figsize=(20,20))

for axi , var in zip(ax.flat,categorical_var_NoChurn):

    sns.countplot(x=df.Churn,hue=df[var],ax=axi)
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

for x in [i for i in df.columns if len(df[i].unique())==2]:

    print(x, df[x].unique())

    df[x]= label_encoder.fit_transform(df[x])
#Check Variables after Encoding

[[x, df[x].unique()] for x in [i for i in df.columns if len(df[i].unique())<10]]
#Encode Variables with more than 2 Classes

df= pd.get_dummies(df, columns= [i for i in df.columns if df[i].dtypes=='object'],drop_first=True)

  
#Check Variables after Encoding

[[x, df[x].unique()] for x in [i for i in df.columns if len(df[i].unique())<10]]
#Create Features DataFrame

X=df.drop('Churn',axis=1)

#Create Target Series

y=df['Churn']

#Split Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#Scale Data

sc= StandardScaler()

X_train = sc.fit_transform(X_train)

X_train=pd.DataFrame(X_train,columns=X.columns)

X_test=sc.transform(X_test)
#Check Data after Scaling

X_train.head()
#Apply RandomForest Algorethm

random_classifier= RandomForestClassifier()

random_classifier.fit(X_train,y_train)
y_pred= random_classifier.predict(X_test)
#Classification Report

print(classification_report(y_test,y_pred))
#Confusion Matrix

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

          xticklabels=['No','Yes'],

          yticklabels=['No','Yes'] )

plt.xlabel('true label')

plt.ylabel('predicted label')
#get features Importances

xx= pd.Series(random_classifier.feature_importances_,index=X.columns)

xx.sort_values(ascending=False)
y_pred_proba=random_classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

roc_auc=auc(fpr,tpr)

#Now Draw ROC using fpr , tpr

plt.plot([0, 1], [0, 1], 'k--',label='Random')

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('Random Forest ROC curve')

plt.legend(loc='best')

svm_classifier= SVC(probability=True)

svm_classifier.fit(X_train,y_train)
#Predict

y_pred_svm= svm_classifier.predict(X_test)

#Classification Report

print(classification_report(y_test,y_pred_svm))
#Confusion Matrix

mat_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(mat_svm.T, square=True, annot=True, fmt='d', cbar=False,

          xticklabels=['No','Yes'],

          yticklabels=['No','Yes'] )

plt.xlabel('true label')

plt.ylabel('predicted label')
y_pred_svm_proba=svm_classifier.predict_proba(X_test)[:,1]

#ROC Curve

fpr_svm, tpr_svm, _svm = roc_curve(y_test, y_pred_svm_proba)

roc_auc=auc(fpr_svm,tpr_svm)

#Now Draw ROC using fpr , tpr

plt.plot([0, 1], [0, 1], 'k--',label='Random')

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('SVM ROC curve')

plt.legend(loc='best')

#Initiate ANN Classifier

ann_classifier= Sequential()

X.shape
#Adding Hidden Layer1

ann_classifier.add(Dense(12,activation='relu',kernel_initializer='uniform',input_dim=23))

#Adding Hidden Layer2

ann_classifier.add(Dense(12,activation='relu',kernel_initializer='uniform'))

#Adding output Layer

ann_classifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

#Compile them Model

ann_classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])
ann_classifier.summary()
%time ann_classifier.fit(X_train,y_train,batch_size=10,epochs=100)
#Get Prediction Proba

y_pred_ann_proba= ann_classifier.predict(X_test)
#Convert Prediction to Int

y_pred_ann= (y_pred_ann_proba>.5).astype('int')
#Priint Classification Report

print(classification_report(y_test,y_pred_ann))
#Confusion Matrix

mat_ann = confusion_matrix(y_test, y_pred_ann)

sns.heatmap(mat_ann.T, square=True, annot=True, fmt='d', cbar=False,

          xticklabels=['No','Yes'],

          yticklabels=['No','Yes'] )

plt.xlabel('true label')

plt.ylabel('predicted label')
#Roc Curve

fpr_ann,tpr_ann,_ann=roc_curve(y_test,y_pred_ann_proba)

roc_auc=auc(fpr_ann,tpr_ann)

#Now Draw ROC using fpr , tpr

plt.plot([0, 1], [0, 1], 'k--',label='Random')

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')