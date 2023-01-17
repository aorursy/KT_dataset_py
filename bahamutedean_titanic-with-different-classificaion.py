# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_path = "../input/"+os.listdir("../input")[0]
test_path = "../input/"+os.listdir("../input")[2]
gender_path = "../input/"+os.listdir("../input")[1]
# Any results you write to the current directory are saved as output.
test_dataset = pd.read_csv( test_path )
train_dataset = pd.read_csv( train_path)
gender_submission = pd.read_csv( gender_path )
# to replace the categorical features into numeric features
def replace_embarked( x ):
    if x=='Q':
        return 2
    elif x=='C':
        return 1
    else:
        return 0


def replace_gender(x):
    if x=='male':
        return 0
    else:
        return 1

def replace_letter( x ):
    try:
        output = float(x.split(' ')[-1])
    except ValueError:
        output=None

    return output

train_dataset['Sex']=train_dataset['Sex'].apply( replace_gender )
train_dataset['Ticket']=train_dataset['Ticket'].apply( replace_letter )
train_dataset['Embarked']=train_dataset['Embarked'].apply( replace_embarked )
# to fill the nan value with the mean 

train_transformed = train_dataset[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']]
mean = train_transformed['Age'].mean()
train_transformed['Age'].fillna( mean,inplace=True )
mean = train_transformed['Ticket'].mean()
train_transformed['Ticket'].fillna( mean,inplace=True )
mean = train_transformed['Fare'].mean()
train_transformed['Fare'].fillna( mean,inplace=True )

test_dataset['Sex']=test_dataset['Sex'].apply( replace_gender )
test_dataset['Ticket']=test_dataset['Ticket'].apply( replace_letter )
test_dataset['Embarked']=test_dataset['Embarked'].apply( replace_embarked )
test_transformed = test_dataset[['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']]
mean = test_transformed['Age'].mean()
test_transformed['Age'].fillna( mean,inplace=True )
mean = test_transformed['Ticket'].mean()
test_transformed['Ticket'].fillna( mean,inplace=True )
mean = test_transformed['Fare'].mean()
test_transformed['Fare'].fillna( mean,inplace=True )
#test_transformed = test_transformed.fillna( test_transformed.mean(),inplace=True )
del mean
from sklearn.preprocessing import MinMaxScaler
MinMax_convertion = MinMaxScaler(copy=False, feature_range=(0, 1))
train_data = train_transformed.values.tolist()

test_data = test_transformed.values.tolist()

print( 'the dataset and label of train and test have been prepared!\n' )


train_data = MinMax_convertion.fit_transform( train_data )
test_data = MinMax_convertion.fit_transform( test_data )
train_labels = train_dataset['Survived'].values.tolist()
test_labels = gender_submission['Survived'].values.tolist()

print( 'the dataset has been min_max scaled!\n' )

from sklearn.metrics import accuracy_score

# with bayes classification

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB( )
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in GaussianNB is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )
# with SVC
from sklearn import svm

clf = svm.SVC( C=0.5,degree=5,max_iter=200 )
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in SVC is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )



clf = svm.LinearSVC (  )
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in Linear_SVC is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )

#with decision tree classification in information gain kernal
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',splitter='random')
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in DecisionTree is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )
# with RF classification
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy',n_jobs=4)
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in RandomForest is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )
# now we could use XGboost classification

from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in XGBoost is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )
# with the KNN 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=25)
clf.fit( train_data,train_labels )
pred = clf.predict( test_data )
print( 'the accuracy rate of Titanic test_dataset in KNN is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
clf.fit(train_data)
pred = clf.predict( test_data )
#print( pred[:10] )
#print( '= ='*40 )
#print( test_labels[:10] )

from sklearn.metrics import adjusted_rand_score
accuracy = adjusted_rand_score( test_labels,pred )
print( 'the Rank_score of Titanic test_dataset in K-means is %.2f' %( accuracy ) )
print( 'the accuracy rate of Titanic test_dataset in K-means is %.2f%%' %( accuracy_score(test_labels,pred)*100 ) )