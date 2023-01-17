# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

import math

from pandas import Series, DataFrame 

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score



# User-defined function

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):

    clf=clf.fit(X_train, y_train)    

    y_pred=clf.predict(X_test)

    score_pred=accuracy_score(y_test, y_pred)

    return score_pred



def choose_classifier(choice):

    if (choice=='Decision Tree'):

        from sklearn import tree

        clf=tree.DecisionTreeClassifier()        

    elif (choice=='Random Forest'):

        from sklearn.ensemble import RandomForestClassifier

        clf= RandomForestClassifier()

    elif (choice=='Logistic Regression'):

        from sklearn.linear_model.logistic import LogisticRegression

        clf=LogisticRegression()        

    elif (choice=='Naive Bayes'):

        from sklearn.naive_bayes import GaussianNB        

        clf= GaussianNB() 

    elif (choice=='Support Vector Machine'):

        from sklearn import svm

        clf=svm.SVC() 

    elif (choice=='K Neighbors'):

        from sklearn.neighbors import KNeighborsClassifier

        clf=KNeighborsClassifier()     

    elif (choice=='Stochastic Gradient Descent'):

        from sklearn.linear_model import SGDClassifier

        clf=SGDClassifier()        

    elif (choice=='Gradient Boosting'):

        from sklearn.ensemble import GradientBoostingClassifier

        clf=GradientBoostingClassifier()

    return clf       



# Uploading data from csv file

df_01=pd.read_csv('../input/train.csv', sep=',')

print('(1)Row and Colmun of raw data： ',df_01.shape,'\n')
# Preprocessing the data

titanic_X = df_01.loc[:,['Pclass','Age','Sex','Fare']]

mean_ages = np.mean(titanic_X.iloc[:, 1]) 

mean_fare = np.mean(titanic_X.iloc[:, 3]) 

(Row_end,Col_end)=titanic_X.shape

for i in range(Row_end):

    for j in range(Col_end):

        if (j==1):

            if (math.isnan(float(titanic_X.ix[i,j]))==True):

                titanic_X.ix[i,j]=mean_ages       

        elif (j==2):

            if (titanic_X.ix[i,j]=='male'):

                titanic_X.ix[i,j]=1

            else:

                titanic_X.ix[i,j]=0

        elif (j==3):

            if (titanic_X.ix[i,j]==0):

                titanic_X.ix[i,j]=mean_fare   

print('(2)Descriptive statistics：\n',titanic_X.describe(),'\n')                

titanic_y = df_01[list(df_01.columns)[1]]

X_train, X_test, y_train, y_test=train_test_split(titanic_X,titanic_y,test_size=0.2,random_state=0)
# Training models and then evaluating their performance

Models=['Decision Tree','Random Forest','Logistic Regression',

        'Naive Bayes','Support Vector Machine','K Neighbors',

        'Stochastic Gradient Descent','Gradient Boosting']    

Accuracy=np.zeros(len(Models))

Results_dict={'Models':Models,'Accuracy':Accuracy}

Results_df=pd.DataFrame(Results_dict,columns=['Models','Accuracy'])

for i in range(0,len(Models)): 

    classifier=choose_classifier(Results_df.ix[i,0])    

    Results_df.ix[i,1]=train_and_evaluate(classifier,X_train,y_train,X_test,y_test)       

print('(3)Results：\n',Results_df,'\n')
# Visualizing results

import matplotlib.pyplot as plt

plt.xlabel("Accuracy(%)")

y=np.arange(len(Models))

x=(Results_df['Accuracy']*100)

plt.barh(y,x,color='gold',align='center',alpha=0.8)

plt.yticks(y, Results_df['Models'])

print('(4)The Bar Chart of results：\n')

plt.show()