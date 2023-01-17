# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import pylab as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

plt.rc('figure', figsize=(10,5))
fizsize_with_subplots = (10,10)
bin_size = 10
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
print(df_train.Embarked.unique())
# Credits for the function - http://nbviewer.jupyter.org/github/donnemartin/data-science-ipython-notebooks/blob/master/kaggle/titanic.ipynb
def clean_data(df, drop_passenger_id):
    
    # Get the unique values of Sex
    sexes = sorted(df['Sex'].unique())
    
    # Generate a mapping of Sex from a string to a number representation    
    genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

    # Transform Sex from a string to a number representation
    df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
    
    # Get the unique values of Embarked
    embarked_locs = df['Embarked'].unique()

    # Generate a mapping of Embarked from a string to a number representation        
    embarked_locs_mapping = dict(zip(embarked_locs, 
                                     range(0, len(embarked_locs) + 1)))
    
    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)
    
    # Fill in missing values of Embarked
    # Since the vast majority of passengers embarked in 'S': 3, 
    # we assign the missing values in Embarked to 'S':
    df.Embarked[df['Embarked'].isnull()] = 'S'
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val' : 
                       { embarked_locs_mapping[nan] : embarked_locs_mapping['S'] 
                       }
                   }, 
                   inplace=True)
    
    # Fill in missing values of Fare with the average Fare
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.  
    # We'll use the median instead of the mean because the Age 
    # histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
                        .groupby([df['Sex_Val'], df['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
            
    # Define a new feature FamilySize that is the sum of 
    # Parch (number of parents or children on board) and 
    # SibSp (number of siblings or spouses):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # Drop the columns we won't use:
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    
    # Drop the Age column since we will be using the AgeFill column instead.
    # Drop the SibSp and Parch columns since we will be using FamilySize.
    # Drop the PassengerId column since it won't be used as a feature.
    df = df.drop(['Age', 'SibSp', 'Parch'], axis=1)
    
    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    
    return df
df_train = clean_data(df_train, drop_passenger_id=True)
df_test = pd.read_csv('../input/test.csv')
df_test = clean_data(df_test, drop_passenger_id= False)

train_features = df_train.values[:, 1:]
train_target = df_train.values[:, 0]
test_features = df_test.values[:,1:]

train_x, test_x, train_y, test_y = train_test_split(train_features, train_target, test_size=0.3, random_state=0)
print (train_features.shape, train_target.shape)
print (train_x.shape, train_y.shape)
print (test_x.shape, test_y.shape)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print('Accuracy - ' +  str(accuracy_score(test_y, predict_y)))
print ('Confusion Matrix - \n')
print (metrics.confusion_matrix(test_y, predict_y))
# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
print('Accuracy - ' +  str(accuracy_score(test_y, predict_y)))
print ('Confusion Matrix - \n')
print (metrics.confusion_matrix(test_y, predict_y))
from sklearn.svm import SVC
clf = SVC(kernel='linear')

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print('Accuracy - ' +  str(accuracy_score(test_y, predict_y)))
print ('Confusion Matrix - \n')
print (metrics.confusion_matrix(test_y, predict_y))
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print('Accuracy - ' +  str(accuracy_score(test_y, predict_y)))
print ('Confusion Matrix - \n')
print (metrics.confusion_matrix(test_y, predict_y))
# KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=10, weights='uniform')
clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print('Accuracy - ' +  str(accuracy_score(test_y, predict_y)))
print ('Confusion Matrix - \n')
print (metrics.confusion_matrix(test_y, predict_y))