# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#get tools

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import time

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import discriminant_analysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, roc_curve, roc_auc_score, classification_report, auc 

df=pd.read_csv('../input/creditcard.csv')

print(df.head(5))
#Check for missing data

print(df.isnull().values.any())
#Great! Don't have any missing data.

#Get a summary

df.describe()
#Check the distribution of Fraud vs Legitimate transactions

FraudShare=round(df['Class'].value_counts()[1]/len(df)*100,2)

print(FraudShare,'% of transactions are fraudulent')
#Highly uneven distribution between classes

#Take a look at what fraudulent cases are like

df_temp=df.loc[df['Class']==1]

df_temp.describe()
#Only 492 fraudulent transactions

#Average fraudulent amount swiped $122

#Half of fraudulent transactions are for amounts less than $10



#Ideally, split data into train and test sets first

#Then run PCA and standardization on train set to avoid information leak

#But host data already has PCA run on V variables over entire data set

#So standardize Amount and Time to "jive"



sc=StandardScaler()

df['Amount_scaled']=sc.fit_transform(df['Amount'].values.reshape(-1,1))

df['Time_scaled']=sc.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'],axis=1,inplace=True)

print(df.head(3))
#Given the highly uneven distribution between classes, need

#(1) a training set with an even distribution of both classes; and

#(2) an unseen test set with a majority to minority class distribution similar to original dataset



#Random undersample the majority class for 1:1 distribution

#First shuffle the data (argument 'frac' specifies fraction of rows to return in random sample)

df.sample(frac=1)



#From description of Fraud set, we know we have 492 Fraud cases

#Get 492 Non-fraud cases from randomly shuffled set

df_Norm=df.loc[df['Class']==0][:492]

df_Fraud=df.loc[df['Class']==1]



#Join dfFraud and dfNorm to get a normally distributed df

df_norm_dist=pd.concat([df_Norm,df_Fraud])



#Shuffle dataframe rows

df_New=df_norm_dist.sample(frac=1,random_state=22)



#Get required non-fraud cases for test set

df_NormUp=df.loc[df['Class']==0][493:58000]
#We don't know what the 'V' variables are and how they affect 'Class'

#How are variables correlated?

#Can we remove non-supporting ones for more parsimonious model?



sub_sample_corr=df_New.corr()

plt.figure(figsize=(10,10))

sns.heatmap(sub_sample_corr,cmap='coolwarm_r')
#Drop uncorrelated, non-supporting variables from the training features

df_New.drop(['V8','V13','V23','V26','V27','V28','Amount_scaled'],axis=1,inplace=True)

print(df_New.head(3))
#Identifying fraud vs. non-fraud cases is a classification problem

#Need to check for extreme outliers in supporting variables

#Plot box and whisker charts to see



#Variables with positive correlation with class

#V2,V4,V11,V19,Time_scaled

f,axes=plt.subplots(ncols=5,figsize=(20,5))



sns.boxplot(x='Class',y='V2',data=df_New,ax=axes[0])

axes[0].set_title('V2 vs Class: Positive Correlation')



sns.boxplot(x='Class',y='V4',data=df_New,ax=axes[1])

axes[1].set_title('V4 vs Class: Positive Correlation')

            

sns.boxplot(x='Class',y='V11',data=df_New,ax=axes[2])

axes[2].set_title('V11 vs Class: Positive Correlation')



sns.boxplot(x='Class',y='V19',data=df_New,ax=axes[3])

axes[3].set_title('V19 vs Class: Positive Correlation')



sns.boxplot(x='Class',y='Time_scaled',data=df_New,ax=axes[4])

axes[4].set_title('Time_scaled vs Class: Positive Correlation')



plt.show()



#Variables with negative correlation with class

#V1,V3,V5,V6,V7,V9,V10,V12,V14,V15,V16,V17,V18

f,axes=plt.subplots(ncols=5,figsize=(20,5))



sns.boxplot(x='Class',y='V1',data=df_New,ax=axes[0])

axes[0].set_title('V1 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V3',data=df_New,ax=axes[1])

axes[1].set_title('V3 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V5',data=df_New,ax=axes[2])

axes[2].set_title('V5 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V6',data=df_New,ax=axes[3])

axes[3].set_title('V6 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V7',data=df_New,ax=axes[4])

axes[4].set_title('V7 vs Class: Negative Correlation')



plt.show()



f,axes=plt.subplots(ncols=5,figsize=(20,5))

sns.boxplot(x='Class',y='V9',data=df_New,ax=axes[0])

axes[0].set_title('V9 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V10',data=df_New,ax=axes[1])

axes[1].set_title('V10 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V12',data=df_New,ax=axes[2])

axes[2].set_title('V12 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V14',data=df_New,ax=axes[3])

axes[3].set_title('V14 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V15',data=df_New,ax=axes[4])

axes[4].set_title('V15 vs Class: Negative Correlation')



plt.show()



f,axes=plt.subplots(ncols=5,figsize=(20,5))



sns.boxplot(x='Class',y='V16',data=df_New,ax=axes[0])

axes[0].set_title('V16 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V17',data=df_New,ax=axes[1])

axes[1].set_title('V17 vs Class: Negative Correlation')



sns.boxplot(x='Class',y='V18',data=df_New,ax=axes[2])

axes[2].set_title('V18 vs Class: Negative Correlation')



plt.show()
#Get rid of variables with collinearity

#Keep an edited set of predictors with fewer outliers

df_New.drop(['V2','V3','V5','V7','V9','V11','V15','V19','V20','V21','V22','V24','V25','Time_scaled'],axis=1,inplace=True)

df_New.head()
#Trim off extreme outliers in predictor variables

#Start with a larger value since don't have a lot of data to work with

#Using a smaller value, e.g. 2.5, might lose too much information



#V1

V1_fraud=df_New['V1'].loc[df_New['Class']==1].values

q_25,q_75=np.percentile(V1_fraud,25),np.percentile(V1_fraud,75)

#Get interquartile range

V1_fraud_iqr=q_75-q_25

#Set cut-off at 2 times interquartile range

V1_fraud_cutoff=V1_fraud_iqr*4.25

V1_fraud_low,V1_fraud_high=q_25-V1_fraud_cutoff,q_75+V1_fraud_cutoff

df_New=df_New.drop(df_New[(df_New['V1'] > V1_fraud_high) | (df_New['V1'] < V1_fraud_low)].index)



#V6

V6_fraud=df_New['V6'].loc[df_New['Class']==1].values

q_25,q_75=np.percentile(V6_fraud,25),np.percentile(V6_fraud,75)

V6_fraud_iqr=q_75-q_25

V6_fraud_cutoff=V6_fraud_iqr*4.25

V6_fraud_low,V6_fraud_high=q_25-V6_fraud_cutoff,q_75+V6_fraud_cutoff

df_New=df_New.drop(df_New[(df_New['V6'] > V6_fraud_high) | (df_New['V6'] < V6_fraud_low)].index)





#V10

V10_fraud=df_New['V10'].loc[df_New['Class']==1].values

q_25,q_75=np.percentile(V10_fraud,25),np.percentile(V10_fraud,75)

V10_fraud_iqr=q_75-q_25

V10_fraud_cutoff=V10_fraud_iqr*4.25

V10_fraud_low,V10_fraud_high=q_25-V10_fraud_cutoff,q_75+V10_fraud_cutoff

df_New=df_New.drop(df_New[(df_New['V10'] > V10_fraud_high) | (df_New['V10'] < V10_fraud_low)].index)



#V12

V12_fraud=df_New['V12'].loc[df_New['Class']==1].values

q_25,q_75=np.percentile(V12_fraud,25),np.percentile(V12_fraud,75)

V12_fraud_iqr=q_75-q_25

V12_fraud_cutoff=V12_fraud_iqr*4.25

V12_fraud_low,V12_fraud_high=q_25-V12_fraud_cutoff,q_75+V12_fraud_cutoff

df_New=df_New.drop(df_New[(df_New['V12'] > V12_fraud_high) | (df_New['V12'] < V12_fraud_low)].index)



#V14

V14_fraud=df_New['V14'].loc[df_New['Class']==1].values

q_25,q_75=np.percentile(V14_fraud,25),np.percentile(V14_fraud,75)

V14_fraud_iqr=q_75-q_25

V14_fraud_cutoff=V14_fraud_iqr*4.25

V14_fraud_low,V14_fraud_high=q_25-V14_fraud_cutoff,q_75+V14_fraud_cutoff

df_New=df_New.drop(df_New[(df_New['V14'] > V14_fraud_high) | (df_New['V14'] < V14_fraud_low)].index)

#Split into training and test sets first.

X=df_New.drop(labels='Class',axis=1) #features

Y=df_New.loc[:,'Class'] #response



X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=22)

#Remember to prep a test set with uneven distribution of classes

#Minority class represented only 0.17% of original

#Therefore, we need to upsample the majority class in the test set

df_NormUp.drop(['V8','V13','V23','V26','V27','V28','Amount_scaled'],axis=1,inplace=True)

df_NormUp.drop(['V2','V3','V5','V7','V9','V11','V15','V19','V20','V21','V22','V24','V25','Time_scaled'],axis=1,inplace=True)

x=df_NormUp.drop(labels='Class',axis=1)

y=df_NormUp.loc[:,'Class']



testx=pd.concat([X_test,x])

testy=pd.concat([Y_test,y])

#Let's fit a dummy classifier to use as a no-skill benchmark



from sklearn.dummy import DummyClassifier

DUM_mod=DummyClassifier(random_state=22)

DUM_mod.fit(X_train,Y_train)

Y_pred_DUM=DUM_mod.predict(testx)

#Fit K Neighbors Classifier, QDA, Logit, and Random Forest Classifier models to data

#Time estimation process



KNC_mod=KNeighborsClassifier()

QDA_mod=discriminant_analysis.QuadraticDiscriminantAnalysis()

LOG_mod=LogisticRegression()

RFC_mod=RandomForestClassifier(max_depth=4,random_state=22)



t0_est=time.time()



KNC_mod.fit(X_train,Y_train)

QDA_mod.fit(X_train,Y_train)

LOG_mod.fit(X_train,Y_train)

RFC_mod.fit(X_train,Y_train)



t1_est=time.time()

print('Estimation of all four models took: {0:.4f} seconds'.format(t1_est-t0_est))
#Predict using test sample

#Time process to see how long each takes



t0_KNC=time.time()

Y_pred_KNC=KNC_mod.predict(testx)

t1_KNC=time.time()

print("Predicting with K Neighbors Classifier model took: {0:.4f} seconds".format(t1_KNC-t0_KNC))



t0_QDA=time.time()

Y_pred_QDA=QDA_mod.predict(testx)

t1_QDA=time.time()

print("Predicting with QDA model took: {0:.4f} seconds".format(t1_QDA-t0_QDA))



t0_LOG=time.time()

Y_pred_LOG=LOG_mod.predict(testx)

t1_LOG=time.time()

print("Predicting with Logistic Classification model took: {0:.4f} seconds".format(t1_LOG-t0_LOG))



t0_RFC=time.time()

Y_pred_RFC=RFC_mod.predict(testx)

t1_RFC=time.time()

print("Predicting with Random Forest Classifier model took: {0:.4f} seconds".format(t1_RFC-t0_RFC))
#Plot precision recall curve



#Calculate precision recall curves

precision_DUM,recall_DUM,threshold_DUM=precision_recall_curve(testy,Y_pred_DUM)

precision_KNC,recall_KNC,threshold_KNC=precision_recall_curve(testy,Y_pred_KNC)

precision_QDA,recall_QDA,threshold_QDA=precision_recall_curve(testy,Y_pred_QDA)

precision_LOG,recall_LOG,threshold_LOG=precision_recall_curve(testy,Y_pred_LOG)

precision_RFC,recall_RFC,threshold_RFC=precision_recall_curve(testy,Y_pred_RFC)



#Calculate area under precision recall curves

auprc_DUM=auc(recall_DUM,precision_DUM)

auprc_KNC=auc(recall_KNC,precision_KNC)

auprc_QDA=auc(recall_QDA,precision_QDA)

auprc_LOG=auc(recall_LOG,precision_LOG)

auprc_RFC=auc(recall_RFC,precision_RFC)



print ("Area under precision recall curve, Dummy Classifier model: {0:.4f}".format(auprc_DUM))

print ("Area under precision recall curve, K Neighbors Classification model: {0:.4f}".format(auprc_KNC))

print ("Area under precision recall curve, QDA model: {0:.4f}".format(auprc_QDA))

print ("Area under precision recall curve, Logistic Classification model: {0:.4f}".format(auprc_LOG))

print ("Area under precision recall curve, Random Forest Classification model: {0:.4f}".format(auprc_RFC))
#Hmm, all have AUPRC<0.5 but outperform dummy classifier



#Check Precision Recall Score

AP_DUM=average_precision_score(testy,Y_pred_DUM)

AP_KNC=average_precision_score(testy,Y_pred_KNC)

AP_QDA=average_precision_score(testy,Y_pred_QDA)

AP_LOG=average_precision_score(testy,Y_pred_LOG)

AP_RFC=average_precision_score(testy,Y_pred_RFC)



print('Average Precision Score, DUM model: {0:.4f}'.format(AP_DUM))

print('Average Precision Score, KNC model: {0:.4f}'.format(AP_KNC))

print('Average Precision Score, QDA model: {0:.4f}'.format(AP_QDA))

print('Average Precision Score, LOG model: {0:.4f}'.format(AP_LOG))

print('Average Precision Score, RFC model: {0:.4f}'.format(AP_RFC))
#Not great. But show some skill relative to the dummy model



#Make a classification report for each model

labels=['No Fraud','Fraud']

print('--'*30)

print('Dummy Classification Model')

print('--'*30)

print(classification_report(testy,Y_pred_DUM,target_names=labels))

print('--'*30)

print('K Neighbors Classification Model')

print('--'*30)

print(classification_report(testy,Y_pred_KNC,target_names=labels))

print('--'*30)

print('Quadratic Discriminant Analysis Model')

print('--'*30)

print(classification_report(testy,Y_pred_QDA,target_names=labels))

print('--'*30)

print('Logistic Regression Model')

print('--'*30)

print(classification_report(testy,Y_pred_LOG,target_names=labels))

print('--'*30)

print('Random Forest Classifier Model')

print('--'*30)

print(classification_report(testy,Y_pred_RFC,target_names=labels))

print('--'*30)
print(confusion_matrix(testy,Y_pred_DUM))

print(confusion_matrix(testy,Y_pred_KNC))

print(confusion_matrix(testy,Y_pred_QDA))

print(confusion_matrix(testy,Y_pred_LOG))

print(confusion_matrix(testy,Y_pred_RFC))