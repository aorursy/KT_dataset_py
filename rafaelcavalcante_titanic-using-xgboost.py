# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time
import csv
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
# Pretty display for notebooks
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#dowloading the images
train_data = pd.read_csv(os.path.join("../input", "train.csv"))
train_data.head()

test_data = pd.read_csv(os.path.join("../input", "test.csv"))
test_data.head()
#now it's time to evaluate the data

train_data.info()

# observe that there are null objects in Age and Cabin in the test size, it is important to evaluate how
#we are going to deal with it, lets compare with the test_set

test_data.info()
#they both have the same behavior, lets create a isnull collumn for each one of them. 
# now lets observe how they are divided
train_data.describe()

test_data.describe()
#now it is time to check how the data is divided in each feature
for column in train_data:
    if (train_data[column].value_counts().shape[0] <10):
        print (column)
        print(train_data[column].value_counts())
#for this analysis, we consider that it's worth to get a deep knowledge in the Sex and embarked as the object data.
#for do this, we gonna plot them

for column in test_data:
    if (test_data[column].value_counts().shape[0] <10):
        print (column)
        print(test_data[column].value_counts())
#From here, we can analyse that Sibsp and Parch has a lot of features of 0 and 1, is it
#important to the output?
# Besides, there is the number 9 in parch, that only exists in test set, I will manually modify it to
#to another existent feature, that has the same output


y_train = train_data["Survived"]
x_train = train_data.drop("Survived", axis = 1)
#now we gonna create the function no analyses the influence of the categorical values in the survival rate 
def SurvivalAnalysis(train_data,key ='Sex', values = ['male', 'female'] ):
# Create DataFrame containing categories and count of each
    frame = pd.DataFrame(index = np.arange(len(values)), columns=(key,'Survived','NSurvived'))
    for i, value in enumerate(values):
        frame.loc[i] = [value, \
                       len(train_data[(train_data['Survived'] == 1) & (train_data[key] == value)]), \
                       len(train_data[(train_data['Survived'] == 0) & (train_data[key] == value)])]
    print(frame.head())

    # Set the width of each bar
    bar_width = 0.4
     # Display each category's survival rates
    for i in np.arange(len(frame)):
        nonsurv_bar = plt.bar(i-bar_width, frame.loc[i]['NSurvived'], width = bar_width, color = 'r')
        surv_bar = plt.bar(i, frame.loc[i]['Survived'], width = bar_width, color = 'g')
        plt.xticks(np.arange(len(frame)), values)
        plt.legend((nonsurv_bar[0], surv_bar[0]),('Did not survive', 'Survived'), framealpha = 0.8)

        # Common attributes for plot formatting
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(train_data[key])):
        nan_outcomes = train_data[pd.isnull(train_data[key])]['Survived']
        print("Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
        key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))

#from here, we can conclude that if you were a woman, it would be more likely youo'd survive        
SurvivalAnalysis(train_data,key ='Sex', values = ['male', 'female'])
SurvivalAnalysis(train_data,key ='Pclass', values =  np.arange(1,4))
SurvivalAnalysis(train_data,key ='Embarked', values =  ['C', 'Q', 'S'])
SurvivalAnalysis(train_data,key ='SibSp', values =  [0,1,2,3,4,5,8])
SurvivalAnalysis(train_data,key ='Parch', values =  [0,1,2,3,4,5,6])


#from here, we can conclude that, male, third class and S are in the worst situation
# sieblings and parch we're gonna use as, 0, or more than 0

#now it is time to analyse the number features ( fare and age)

def AnalyseNumeric (train_data, key):
   # Remove NaN values from Age data - Remember to include a collumn is null
    train_data = train_data[~np.isnan(train_data[key])]
    # Divide the range of data into bins and count survival rates
    min_value = train_data[key].min()
    max_value = train_data[key].max()
    value_range = max_value - min_value
    # 'Fares' has larger range of values than 'Age' so create more bins
    if(key == 'Fare'):
        bins = np.arange(0, train_data['Fare'].max() + 20, 20)
    if(key == 'Age'):
        bins = np.arange(0, train_data['Age'].max() + 10, 10)
        
    # Overlay each bin's survival rates
    nonsurv_vals = train_data[train_data['Survived'] == 0][key].reset_index(drop = True)
    surv_vals = train_data[train_data['Survived'] == 1][key].reset_index(drop = True)
    plt.hist(nonsurv_vals, bins = bins, alpha = 0.6,
    color = 'red', label = 'Did not survive')
    plt.hist(surv_vals, bins = bins, alpha = 0.6,
    color = 'green', label = 'Survived')
    
        # Add legend to plot
    plt.xlim(0, bins.max())
    plt.legend(framealpha = 0.8)
    
    plt.xlabel(key)
    plt.ylabel('Number of Passengers')
    plt.title('Passenger Survival Statistics With \'%s\' Feature'%(key))
    plt.show()

    # Report number of passengers with missing values
    if sum(pd.isnull(train_data[key])):
        nan_outcomes = train_data[pd.isnull(train_data[key])]['Survived']
        print("Passengers with missing '{}' values: {} ({} survived, {} did not survive)".format( \
              key, len(nan_outcomes), sum(nan_outcomes == 1), sum(nan_outcomes == 0)))

AnalyseNumeric (train_data, "Fare")
AnalyseNumeric (train_data, "Age")
#now it's time to select the data
test_data.head()
#let´s get rid from nan values from age
x_train.fillna(-999, inplace = True)
#Now lets create a column that indicates Age was completed
x_train['AgeNull'] = (x_train['Age'] == -999)
x_train['CabinNUll'] = (x_train['Cabin'] == -999)

#let´s get rid from nan values from age and fare
test_data.loc[np.isnan(test_data["Fare"]),"Fare"] = test_data["Fare"].median()
test_data.fillna(-999, inplace = True)
#Now lets create a column that indicates Age was completed
test_data['AgeNull'] = (test_data['Age'] == -999)
test_data['CabinNUll'] = (test_data['Cabin'] == -999)
test_data.loc[test_data.Parch == 9,"Parch"] = 6
test_data.describe()
#Next we gonna separate only the varibales that we really need in model,
x_trainID = x_train["PassengerId"]
x_testID = test_data["PassengerId"]
x_train = x_train.drop(labels=['Name', 'Ticket', "PassengerId", "Cabin"], axis=1)
x_test = test_data.drop(labels=['Name', 'Ticket', "PassengerId", "Cabin"], axis=1)
x_test.info()
x_train.info()

x_train = pd.get_dummies(x_train)
x_train.describe()
x_test = pd.get_dummies(x_test)
x_test.head()
x_train = x_train.drop(labels=['Embarked_-999'], axis=1)
clf = XGBClassifier()

scorer = make_scorer(f1_score)

clf.fit(x_train, y_train)

# Make predictions using the new model.
best_train_predictions = clf.predict(x_train)

# Calculate the f1_score of the new model.
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print(accuracy_score(y_train, best_train_predictions))



# Calculate the f1_score of the new model.
print('The training F1 Score is', f1_score(best_train_predictions, y_train))
print(accuracy_score(y_train, best_train_predictions))


pred = clf.predict(x_test)



#now it´s time to calculate our outputs

pred = pred.tolist()
passenger = np.array(x_testID).tolist()

with open('gender_submission.csv','w', newline='') as file:
    w = csv.writer(file)
    w.writerow(['PassengerId', 'Survived'])
    w.writerows(zip(passenger, pred))

