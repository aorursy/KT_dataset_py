# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.
#read raw data, exclude unnecessary columns like Name and Ticket and assign PassengerId as the Index.

#this will make the reading process faster. We will also drop the Fair, Cabin columns since these appear

#as noise.



data = pd.read_csv("../input/train.csv", index_col=0, usecols=[0,1,2,4,5,6,7,11])

data.head()
#For each of the columns, now deal with the na values



#Since most passengers embarked at the Southampton port in U.K

data['Embarked'].fillna('S', inplace=True)



#Age is also missing for a lot of entries. We will attempt to fill the age as follows: 

#If there is a parent, we will use the median age of the entries with parent. Similar logic

#for entries which do not have parents



data.loc[data['Parch'] == 0, 'Age'] = data[data['Parch'] == 0]['Age'].fillna(data[data['Parch'] == 0]['Age'].median())

data.loc[data['Parch'] > 0, 'Age'] = data[data['Parch'] > 0]['Age'].fillna(data[data['Parch'] > 0]['Age'].median())



#Convert string values to floats

data.replace(to_replace={'Sex': {'male': 1,'female': 2}, 'Embarked': {'S': 1,'C': 2,'Q': 3}} , inplace = True)

data.head()
#Random Forests

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics



predictors = data.drop(['Survived'], 1)

targets = data.Survived



pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, train_size=0.75)



#Build model on training data

classifier=RandomForestClassifier(n_estimators=25)

classifier=classifier.fit(pred_train,tar_train)



predictions=classifier.predict(pred_test)



#Confusion Matrix - https://en.wikipedia.org/wiki/Confusion_matrix

print ('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(tar_test,predictions))

print ('Accuracy Score is: ', sklearn.metrics.accuracy_score(tar_test, predictions))
# Variable Importance indicates what factors seem to govern survival

from sklearn.ensemble import ExtraTreesClassifier



# fit an Extra Trees model to the data

model = ExtraTreesClassifier(n_estimators=25)

model.fit(pred_train,tar_train)



# constuct dataframe to sort values so we can print out the variables

df = pd.DataFrame(data=model.feature_importances_, index=predictors.columns, columns=['Score'])



df.sort_values(axis=0, ascending=False, by='Score', inplace=True)



df
import matplotlib.pylab as plt



#Running a different number of trees and see the effect

#of that on the accuracy of the prediction. Note that the optimal number of trees changes

#as we run this several times. We should use the classifier we get



trees=range(25)

accuracy=np.zeros(25)

classifiers = np.ndarray(25, dtype=object)



for idx in range(len(trees)):

   classifiers[idx]=RandomForestClassifier(n_estimators=idx + 1)

   classifiers[idx]=classifiers[idx].fit(pred_train,tar_train)

   predictions=classifiers[idx].predict(pred_test)

   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)

   

plt.cla()

plt.plot(trees, accuracy)



print('Maximum Accuracy of {} achieved by using {} Trees'.format(accuracy.max(), accuracy.argmax()+1))