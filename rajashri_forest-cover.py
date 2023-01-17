#Final submission score is 0.67,which needs to be improved.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.svm import SVC

from statistics import variance

from sklearn.feature_selection import VarianceThreshold









print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
train.sample()

train.head()
#check for missing values

train.info()
sns.heatmap(train.isnull(),cbar = False)
distance=pd.DataFrame(train,columns = ['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',

                                      'Hillshade_Noon','Horizontal_Distance_To_Fire_Points'])







for column in distance:

    plt.figure()

    distance.boxplot([column])
#Soil_Type7,Soil_Type15 has 0 standard deviation

#train = train.drop(["Soil_Type7","Soil_Type15"],axis = 1)

#test = test.drop(["Soil_Type7","Soil_Type15"],axis = 1)
#Cover type is the target to be predicted.

#Train test split

x_train,x_test,y_train,y_test=  train_test_split(train.drop('Cover_Type',axis = 1),train['Cover_Type'],test_size = 0.3,random_state = 17)
#Building logistic regression model

logreg = LogisticRegression()

logreg.fit(x_train,y_train)
#Predicting logistic regression results

logreg.predict(x_test)
#Logistic regression test scores

score = logreg.score(x_test, y_test)

print(score)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()

ensemble_model = RandomForestClassifier()

tree_model.fit(x_train,y_train)
ensemble_model.fit(x_train,y_train)
tree_predict=tree_model.predict(x_test)

tree_model.score(x_test,y_test)
ensemble_predict= ensemble_model.predict(test)

print (ensemble_predict)

ensemble_model.score(x_test,y_test)
#of the 3 algorithms applied to the dataset,ensemble model works better with a score of 0.84

submission.shape
x_test.shape

test.shape

#current public score is 0.66,this should be improved

#checking the variance of each feature

train1 = train

test1 = test

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

sel.fit_transform(train1)



train1.head(40)



#this is based on this article ,https://scikit-learn.org/stable/modules/feature_selection.html

#could see no rows being removed in the data set,as all of them have valid values,non null.

tree_model.fit
tree_model.predict(test1)

tree_model.score(x_test,y_test)
pd.DataFrame([train.mean(), train.std(), train.var()], index=['Mean', 'Std. dev', 'Variance'])

#Getting feature importance after running the data through a ensemble model classifier
x=pd.DataFrame(ensemble_model.feature_importances_,

             index=x_train.columns, columns=['Importance']).sort_values(

    by='Importance', ascending=False)[:10]

print(x)
#Modelling based on important features alone

train2 = train

test2 = test

train_imp = train2[['Id','Elevation','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',

                    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am',

                    'Aspect','Hillshade_3pm',

                    'Wilderness_Area4','Cover_Type']]



x_train_imp,x_test_imp,y_train_imp,y_test_imp=  train_test_split(train_imp.drop('Cover_Type',axis = 1),train_imp['Cover_Type'],

                                                                 test_size = 0.3,random_state = 17)

logreg1 = LogisticRegression()

logreg1.fit(x_train_imp,y_train_imp)

logreg1.predict(x_test_imp)

logreg1.score(x_test_imp,y_test_imp)
tree_model1 =DecisionTreeClassifier()

tree_model1.fit(x_train_imp,y_train_imp)

tree_predict=tree_model1.predict(x_test_imp)

tree_model1.score(x_test_imp,y_test_imp)
ensemble_1 = RandomForestClassifier()

ensemble_1.fit(x_train_imp,y_train_imp)

ensemble_predict= ensemble_1.predict(x_test_imp)

print (ensemble_predict)

ensemble_1.score(x_test_imp,y_test_imp)
pd.DataFrame(tree_model.feature_importances_,index = x_train.columns,columns=['Importance']).sort_values(

    by = 'Importance',ascending = False)[:10]
#Both the models have same important features,also these important features completely ignores the soil type feature.

#This should be included,except Soil_Type7,Soil_Type15 has lower standard deviation.



#Modelling based on important features alone

train = train.drop(["Soil_Type7","Soil_Type15","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Slope",

                  "Hillshade_Noon"],axis = 1)

test = test.drop(["Soil_Type7","Soil_Type15","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Slope",

                  "Hillshade_Noon"],axis = 1)

train3 = train

test3 = test



train[:10]

#train_imp = train3[train]



#[['Id','Elevation','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',

 #                   'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am',

  #                  'Aspect','Hillshade_3pm',

   #                 'Wilderness_Area4','Cover_Type']]
x_train_3,x_test_3,y_train_3,y_test_3=  train_test_split(train3.drop('Cover_Type',axis = 1),train3['Cover_Type'],

                                                                 test_size = 0.3,random_state = 17)

logreg2 = LogisticRegression()

logreg2.fit(x_train_3,y_train_3)

logreg2.predict(x_test_3)

logreg2.score(x_test_3,y_test_3)
tree_model2 =DecisionTreeClassifier()

tree_model2.fit(x_train_3,y_train_3)

tree_predict2=tree_model2.predict(x_test_3)

tree_model2.score(x_test_3,y_test_3)

tree_test_pred = tree_model2.predict(test)
ensemble_2 = RandomForestClassifier()

ensemble_2.fit(x_train_3,y_train_3)

ensemble_predict2= ensemble_2.predict(x_test_3)

print (ensemble_predict2)

ensemble_2.score(x_test_3,y_test_3)

ensemble_test_pred = ensemble_2.predict(test)

#Naive Bayes Model
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train_3,y_train_3)

nb.predict(x_test_3)

nb.score(x_test_3,y_test_3)
#SGDClassifier,very low performance
sgd = SGDClassifier(loss = 'modified_huber',shuffle = True,random_state = 171)

sgd.fit(x_train_3,y_train_3)

sgd.predict(x_train_3)

sgd.score(x_test_3,y_test_3)
sgd = SGDClassifier(loss = 'log',shuffle = True,random_state = 171)

sgd.fit(x_train_3,y_train_3)

sgd.predict(x_train_3)

sgd.score(x_test_3,y_test_3)
sgd = SGDClassifier(shuffle = True,random_state = 171)

sgd.fit(x_train_3,y_train_3)

sgd.predict(x_train_3)

sgd.score(x_test_3,y_test_3)
submission = pd.DataFrame({'Id':test.Id,'Cover_Type':ensemble_test_pred})

submission.head()

submission.to_csv('submission.csv',index = False)
submission_tree = pd.DataFrame({'Id':test.Id,'Cover_Type':tree_test_pred})

submission_tree.head()

submission_tree.to_csv('submission2.csv',index = False)
#Extra tree classifier is a tree based model for classification problems

et = ExtraTreeClassifier()

et.fit(x_train_3,y_train_3)

et.predict(x_train_3)

et.score(x_test_3,y_test_3)
from sklearn.semi_supervised import LabelPropagation

lb = LabelPropagation()

lb.fit(x_train_3,y_train_3)

lb.predict(x_train_3)

lb.score(x_test_3,y_test_3)
from sklearn.neighbors import KNeighborsClassifier

knng =KNeighborsClassifier()

knng.fit(x_train_3,y_train_3)

knng.predict(x_train_3)

knng.score(x_test_3,y_test_3)