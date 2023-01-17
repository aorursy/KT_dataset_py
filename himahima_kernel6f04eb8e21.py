import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

%matplotlib inline

data = pd.read_csv("../input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data.head()

data['class'].value_counts()

data.columns.values

data.info()

data.describe()

sns.heatmap(data.corr())

data.corr()
data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)

data.head()
def change_category_to_number(classCat):

    if classCat=='STAR':

        return 0

    elif classCat=='GALAXY':

        return 1

    else:

        return 2
data['classCat'] = data['class'].apply(change_category_to_number)
data.head()

data.drop(['class'],axis=1,inplace=True)

data.head()
X = data.drop('classCat', axis=1)

y = data['classCat']
#Standard Scaler for Data

from sklearn.preprocessing import StandardScaler





scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

#Splitting data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Import Libraries

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

#----------------------------------------------------



#----------------------------------------------------

#Applying VotingClassifier Model 



'''

#ensemble.VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)

'''



#loading models for Voting Classifier

LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=33)

GBCModel_ = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1, random_state=0)

DTModel_ = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state = 33)

RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=1, random_state=33)

KNNModel_ = KNeighborsClassifier(n_neighbors= 4 , weights ='uniform', algorithm='auto')



#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[('LRModel',LRModel_),('GBCModel',GBCModel_),('DTModel',DTModel_),('RFModel',RFModel_),('KNNModel',KNNModel_)], voting='hard')

VotingClassifierModel.fit(X_train, y_train)



#Calculating Details

print('VotingClassifierModel Train Score is : ' , VotingClassifierModel.score(X_train, y_train))

print('VotingClassifierModel Test Score is : ' , VotingClassifierModel.score(X_test, y_test))

print('----------------------------------------------------')



#Calculating Prediction

y_pred = VotingClassifierModel.predict(X_test)

print('Predicted Value for VotingClassifierModel is : ' , y_pred[:10])
#Import Libraries

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

#----------------------------------------------------



#----------------------------------------------------

#Calculating Confusion Matrix

CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)



# drawing confusion matrix

sns.heatmap(CM, center = True)

plt.show()



#----------------------------------------------------

#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

AccScore = accuracy_score(y_test, y_pred, normalize=False)

print('Accuracy Score is : ', AccScore)