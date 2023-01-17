#Load the packages that we will use

import pandas as pd

import numpy as np

import csv

from sklearn import ensemble

from sklearn import tree

import matplotlib.pyplot as plt

import seaborn as sns
import os





#Change it if not conveninent

os.chdir('/kaggle/input')





#Verify it has been changed successfully

os.getcwd()

train_df = pd.read_csv('train.csv', header=0)
# female = 0, Male = 1

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
len(train_df.Embarked[ train_df.Embarked.isnull() ])
from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))
from subprocess import check_output

print(check_output(["ls", "/kaggle/input"]))
os.chdir('/kaggle/input')

os.getcwd()
train_df = pd.read_csv('train.csv', header=0)
whos
train_df.shape
train_df.info()
train_df.describe().transpose()
train_df.dtypes
train_df.head(5)
train_df.tail(5)
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
len(train_df.Embarked[ train_df.Embarked.isnull() ])
median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
mode_embark = train_df['Embarked'].dropna().mode().values



if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.loc[ (train_df.Embarked.isnull()),'Embarked' ] = mode_embark
Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,

Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        train_df.loc[ (train_df.Fare.isnull()) & (train_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
# Data cleanup

# TEST DATA

test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe



# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# All the ages with no data -> make the median of all Ages

median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

    

# All missing Embarked -> just make them embark from most common place

mode_embark = test_df['Embarked'].dropna().mode().values

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.loc[ (test_df.Embarked.isnull()),'Embarked' ] = mode_embark



# Again convert all Embarked strings to int

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)



# All the missing Fares -> assume median of their respective class

if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]



# Collect the test data's PassengerIds before dropping it

ids = test_df['PassengerId'].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
from sklearn.cross_validation import train_test_split



# Convert back to a numpy array and separate the predictors

X = train_df.values[0::, 1::]

y = train_df.values[0::, 0]



# shuffle and split training and test sets

train_data, test_data, train_data_y, test_data_y = train_test_split(X, y, test_size=.25)
print ('Training...')

forest = ensemble.RandomForestClassifier(n_estimators=10000)

forest = forest.fit( train_data, train_data_y )



print ('Predicting...')

output = forest.predict(test_df).astype(int)



predictions_file = pd.DataFrame({'PassengerId':ids, 'Survived':output})

print ('Done.')
#Extracting the features list

features_list = train_df.columns.values[1::]



#Extracting the feature importances

feature_importance = forest.feature_importances_



# Obtaining the importance indexes for the features

sorted_idx = np.argsort(feature_importance)[::-1]



# Comparing former order of features with the ordered one

print("\nFeatures original order:\n", features_list)

print("\nFeatures sorted by importance (DESC):\n", features_list[sorted_idx])



# Calculating the standard deviation of the importances

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)



import matplotlib.ticker as mtick



#Plotting the graph

pos = np.arange(sorted_idx.shape[0]) + .5



plt.subplot(2, 1, 1)

plt.barh(pos, feature_importance[sorted_idx[::-1]],xerr=std[sorted_idx[::-1]],color="g",align='center')

plt.yticks(pos, features_list[sorted_idx[::-1]])



plt.xlabel('Variable Importance')

plt.title('Variable Importance')

plt.draw()

plt.show()

from sklearn.metrics import roc_curve, auc



# Determine the false positive and true positive rates

fpr, tpr, _ = roc_curve(test_data_y, forest.predict_proba(test_data)[:,1])



# Calculate the AUC

roc_auc = auc(fpr, tpr)

print('ROC AUC: %0.2f' % roc_auc)



# Plot of a ROC curve for a specific class

plt.figure()

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()