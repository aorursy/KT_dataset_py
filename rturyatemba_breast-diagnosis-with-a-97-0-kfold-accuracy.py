



#AIM OF THIS KERNEL IS TO TRY DIFFERENT LEARNING AND PREDICTIVE MODELS 

#TEST THIS MODELS WITH THE CROSS VALIDATION CLASSIFIER ON THE DATA PROVIDED

#TO CHOOSE THE MODEL WITH THE HIGHEST LEVEL OF ACCURACY



#we are going to chose multiple model to predict the diagnosis predicting using random values in the 30 features

#so we test which model is the most accurate using the metrics accuracy score class from sklearn



#in this kernel, i am using  the data provided by the dataset  found via https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29



#i am going to use multiple predictive model and choose the most optimum using an accuracy metric of scikit-learn to figure out the best model to use

#that gives us the highest level of accuracy



#first, let us prepare the data

import pandas as pd

import numpy as np



#read data from a csv file

df = pd.read_csv('../input/data.csv',header=0)

#drop this field because it is of no use to us since it has null values all throught

df = df.drop(['Unnamed: 32'], axis=1)



#change the diagnosis values into ints that will be used by the models

df['diagnosis'] = df['diagnosis'].map({'M':0,'B':1}).astype(int)



#print to check

df['diagnosis']



#then slipt the dataset into x and y where x is the 30 features and y is the diagnosis column

#we dont need to convert this diagnosis column into a numeric array,since fit method accepts dataseries

#though if we are to feed in the predict method, an array or list would be required to make a predictions

#so if at any point we need to predict anything, we shall use the numpy library

y= df['diagnosis']



#Then create a dataseries x that has all the other 30 features by dropping the other two columns

x = df.drop(['diagnosis','id'],axis=1)



#import libraries to use for learning and predictability

from sklearn import tree

from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree







#using the cross validation algorithm to test its accuracy

#t_size is the percentage of the dataframe that is in the test ie 0.4 of the number of values 

#train is for training the model and test is for testing the model to predict something in the dataframe

X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=.4,random_state=0)

KFold = KFold(len(df),n_folds=10,shuffle=False)



#using the decision tree classifier

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

print('the accuracy score of a decisiontree classifier is %s' %cross_val_score(Tree,x,y,cv=10).mean())



#try the randomforestclassifier model and see the accuracy

Forest = RandomForestClassifier(n_estimators =10)

Forest = Forest.fit(x,y)

print('the accuracy score of a randomforest classifier is %s' %cross_val_score(Forest,x,y,cv=10).mean())



#using the gradient boosting & adaboost  algorithm

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1,random_state=0)

GBC = GBC.fit(x,y)

print('the accuracy score of the gradient boosting algorithm is %s' %cross_val_score(GBC,x,y,cv=10).mean())



#from the scores, gradient boosting algorithms makes the best classifier to use

#we are going to create random values for each 30 features and predict the diagnosis 

#whether malignant M or benign B using gradient boosting algorithm

print('predicting...')

output = GBC.predict([16.02,23.24,102.7,797.8,0.08206,0.06669,0.03299,0.03323,0.1528,0.05697,0.3795,1.187,2.466,40.51,0.004029,0.009269,0.01101,0.00759,0.0146,0.003042,19.19,33.88,123.8,1150,0.1181,0.1551,0.1459,0.09975,0.2948,0.08452])

print('done')

print(output)


