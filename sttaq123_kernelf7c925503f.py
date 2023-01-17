"""

Created on Wed Mar 20 22:32:51 2019



@author: Wajeeha

"""



import pandas as pd

import os

import matplotlib.pyplot as plt # plotting

import numpy as np

os.getcwd()

os.chdir(r'../input')



df1=pd.read_csv('OLX_Car_Data_CSV.csv', delimiter=',', encoding = "ISO-8859-1")

df1.head(5) # to see the first 5 rows of dataset

df1.shape 

columnNames = pd.Series(df1.columns.values) #get the column names of the dataset

description= df1.describe(include='all') # get the description summary of all columns



#frequency distribution of categorical data

Brand_vc=df1['Brand'].value_counts()

Condition_vc= df1['Condition'].value_counts()

Fuel_vc=df1['Fuel'].value_counts()

Model_vc=df1['Model'].value_counts()

RegisteredCity_vc= df1['Registered City'].value_counts()

TransactionType_vc= df1['Transaction Type'].value_counts()
#Data preprocessing

#Handling missing values

# number of missing values in each column as isnull() returns 1, if the value is null.

MissingData=df1.apply(lambda x: sum(x.isnull()),axis=0) #axis=0 for columns and axis=1 for rows



#dropna

df1.dropna(inplace=True)



#checking updated dataframe

df1.head(5)
#import Normalisation package

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

cols_scaled = sc.fit_transform(df1.loc[:,['KMs Driven','Price']])



# take copy of df1 for scaling

df1_scaled= df1.copy() 

df1_scaled['KMs Driven'] = pd.Series(cols_scaled[:,0]) 

df1_scaled['Price'] = pd.Series(cols_scaled[:,1]) 

df1_scaled.head(5)
#frequency distribution of categorical data

Brand_vc=df1_scaled['Brand'].value_counts()

Condition_vc= df1_scaled['Condition'].value_counts()

Fuel_vc=df1_scaled['Fuel'].value_counts()

Model_vc=df1_scaled['Model'].value_counts()

RegisteredCity_vc= df1_scaled['Registered City'].value_counts()

TransactionType_vc= df1_scaled['Transaction Type'].value_counts()
#creating sub dataframe df1_bdmodfiltered

df1_model=Model_vc[Model_vc >= 100]

df1_modfiltered=df1_scaled.loc[df1_scaled['Model'].isin(df1_model.index)] #select rows whose column value is in or matches index ofdf1_model.index

Brand_vc=df1_modfiltered['Brand'].value_counts()

df1_Brand=Brand_vc[Brand_vc >=100]

df1_bdmodfiltered=df1_modfiltered.loc[df1_modfiltered['Brand'].isin(df1_Brand.index)]

#Brand1_vc=df1_bdmodfiltered['Brand'].value_counts() to check if the brand count again

Brand_vc=df1_bdmodfiltered['Brand'].value_counts()

Model_vc=df1_bdmodfiltered['Model'].value_counts()

df1_bdmodfiltered.head(5)
# Remove leasing and keep Karachi data

df1_clean = df1_bdmodfiltered[df1_bdmodfiltered['Transaction Type'] != 'Installment/Leasing']

df1_clean = df1_clean.loc[df1_clean['Registered City'] == 'Karachi']



TransactionType_vc= df1_clean['Transaction Type'].value_counts()

RegisteredCity_vc= df1_clean['Registered City'].value_counts()



# drop registered city

df1_clean = df1_clean.drop(['Registered City'], axis=1)

df1_clean = df1_clean.drop(['Transaction Type'], axis=1)



df1_clean.Price

df1_clean['KMs Driven']



df1_clean.dropna(inplace=True)

df1_clean.describe()



df1_clean.head(5)



df_num = df1_clean.copy()
#Encoding

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_num['Fuel'] = encoder.fit_transform(df_num['Fuel'])

# save the mapping for future reference

fuelMapping = encoder.classes_

df_num['Brand'] = encoder.fit_transform(df_num['Brand'])

brandMapping = encoder.classes_

df_num['Model'] = encoder.fit_transform(df_num['Model'])

modelMapping = encoder.classes_



df_num.Condition = df_num.Condition.map({'Used':0,'New':1})

df_num.head(5)
import seaborn as sns

corr = df_num.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.pairplot(df_num) #using seaborn pairplots to visualize each feature's skewness - 
#Building models

X= df_num.drop(['Condition'],axis=1)#drop y from the dataframe -  x part will be everything but not the target y variable

y= df_num.Condition



#import classifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

classifier = LogisticRegression()



MissingData2=df_num.apply(lambda x: sum(x.isnull()),axis=0)



# feature extraction

model = LogisticRegression()

rfe = RFE(model, 2)

fit = rfe.fit(X, y)

no_of_features = fit.n_features_

support_features = fit.support_

ranking_features = fit.ranking_

ranking_features



print("Num Features: %d" % (no_of_features))

print("Selected Features: %s" % (support_features))

print("Feature Ranking: %s" % (ranking_features))



X_sub = X.iloc[:,[2,4]]



#split train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)



#train classifier

classifier.fit(X_train,y_train)



#classifier performance on test set

classifier.score(X_test,y_test)



#predictions for test

y_pred = classifier.predict(X_test)
#import performance measure tools

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score 

cm=confusion_matrix(y_test,y_pred)

acs=accuracy_score(y_test,y_pred)

rs=recall_score(y_test,y_pred, average='macro')

ps=precision_score(y_test,y_pred, average='macro')
y_pred_prob = classifier.predict_proba(X_test)[:,1] #

from sklearn.metrics import roc_curve, roc_auc_score

fpr,tpr,thresholds= roc_curve(y_test,y_pred_prob)

#roc curve will return what are the FPR and TPR for different thresholds -  it needs the actual prob so we give predict_prob values instead of predicted values

import matplotlib.pyplot as plt

plt.plot(fpr,tpr)

plt.xlabel("FPR")

plt.ylabel("TPR")

roc_auc_score(y_test,y_pred_prob)