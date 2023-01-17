import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/datasets-for-churn-telecom/cell2celltrain.csv")

test = pd.read_csv("../input/datasets-for-churn-telecom/cell2cellholdout.csv")
train.info()

train[0:10]
#Churn : Yes:1 , No:0

Churn = {'Yes': 1,'No': 0} 

  

# traversing through dataframe 

# values where key matches 

train.Churn = [Churn[item] for item in train.Churn] 

print(train)
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
# for column

#train['MonthlyRevenue'].fillna((train['MonthlyRevenue'].median()), inplace=True)

# for column

train['MonthlyRevenue'] = train['MonthlyRevenue'].replace(np.nan, 0)



# for whole dataframe

train = train.replace(np.nan, 0)



# inplace

train.replace(np.nan, 0, inplace=True)



print(train)



# for column

#train['MonthlyMinutes'].fillna((train['MonthlyMinutes'].median()), inplace=True)

train['MonthlyMinutes'] = train['MonthlyMinutes'].replace(np.nan, 0)



# for whole dataframe

train = train.replace(np.nan, 0)



# inplace

train.replace(np.nan, 0, inplace=True)



print(train)
# for column

#train['TotalRecurringCharge'].fillna((train['TotalRecurringCharge'].median()), inplace=True)

train['TotalRecurringCharge'] = train['TotalRecurringCharge'].replace(np.nan, 0)



# for whole dataframe

train = train.replace(np.nan, 0)



# inplace

train.replace(np.nan, 0, inplace=True)



print(train)
# for column

#train['DirectorAssistedCalls'].fillna((train['DirectorAssistedCalls'].median()), inplace=True)

train['DirectorAssistedCalls'] = train['DirectorAssistedCalls'].replace(np.nan, 0)



# for whole dataframe

train = train.replace(np.nan, 0)



# inplace

train.replace(np.nan, 0, inplace=True)



print(train)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df
train = FunLabelEncoder(train)

train.info()

train.iloc[235:300,:]
test = FunLabelEncoder(test)

test.info()

test.iloc[235:300,:]
test = test.drop(columns=['Churn'],



                 axis=1)

test = test.dropna(how='any')

print(test.shape)
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["Churn"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
# Distribution of Churn

train.Churn.value_counts()[0:30].plot(kind='bar')

plt.show()
train = train[['CustomerID','MonthlyRevenue','MonthlyMinutes','TotalRecurringCharge','DirectorAssistedCalls','OverageMinutes',

         'RoamingCalls','PercChangeMinutes','PercChangeRevenues','DroppedCalls','BlockedCalls','UnansweredCalls','CustomerCareCalls',

         'ThreewayCalls','ReceivedCalls','OutboundCalls','InboundCalls','PeakCallsInOut','OffPeakCallsInOut','DroppedBlockedCalls','CallForwardingCalls'

         ,'CallWaitingCalls','MonthsInService','UniqueSubs','ActiveSubs','ServiceArea','Handsets','HandsetModels',              

'CurrentEquipmentDays','AgeHH1','AgeHH2','ChildrenInHH','HandsetRefurbished','HandsetWebCapable','TruckOwner','RVOwner','Homeownership','BuysViaMailOrder','RespondsToMailOffers','OptOutMailings',          

'NonUSTravel','OwnsComputer','HasCreditCard','RetentionCalls','RetentionOffersAccepted','NewCellphoneUser',          

'NotNewCellphoneUser','ReferralsMadeBySubscriber','IncomeGroup','OwnsMotorcycle','AdjustmentsToCreditRating', 

'HandsetPrice','MadeCallToRetentionTeam','CreditRating','PrizmCode','Occupation','MaritalStatus','Churn']] #Subsetting the data

cor = train.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
from sklearn.model_selection import train_test_split

Y = train['Churn']

X = train.drop(columns=['Churn'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)



# We train model

rfcla.fit(X_train, Y_train)



# We predict target values

Y_predict5 = rfcla.predict(X_test)
# The confusion matrix

rfcla_cm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_rfcla = rfcla.score(X_test, Y_test)

print(score_rfcla)
from sklearn.naive_bayes import GaussianNB



# We define the model

nbcla = GaussianNB()



# We train model

nbcla.fit(X_train, Y_train)



# We predict target values

Y_predict3 = nbcla.predict(X_test)
# The confusion matrix

nbcla_cm = confusion_matrix(Y_test, Y_predict3)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(nbcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('Naive Bayes Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
# Test score

score_nbcla = nbcla.score(X_test, Y_test)

print(score_nbcla)
Testscores = pd.Series([score_rfcla,score_nbcla, ], 

                        index=['Random Forest Score','Naive Bayes Score' ]) 

print(Testscores)
from sklearn.metrics import roc_curve

# Random Forest Classification

Y_predict5_proba = rfcla.predict_proba(X_test)

Y_predict5_proba = Y_predict5_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict5_proba)

plt.subplot(331)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Random Forest')

plt.grid(True)

plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=1.4, hspace=0.45, wspace=0.45)

plt.show()



# Naive Bayes Classification

Y_predict3_proba = nbcla.predict_proba(X_test)

Y_predict3_proba = Y_predict3_proba[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict3_proba)

plt.subplot(332)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='ANN')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('ROC Curve Naive Bayes')

plt.grid(True)

plt.subplots_adjust(top=2, bottom=0.08, left=0.10, right=1.4, hspace=0.45, wspace=0.45)

plt.show()