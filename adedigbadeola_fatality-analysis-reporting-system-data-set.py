# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#I TRIED WORKING ON THIS FIRST
#START
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
fatality_path = '../input/fars_train.csv'
fatality_data = pd.read_csv(fatality_path)
for df in [fatality_data]:
    df['Sex_binary']=df['SEX'].map({'male':1,'female':0})
    df['INJURY_SEVERITY']=df['INJURY_SEVERITY'].map({'Possible_Injury': 0,'No_Injury': 1,'Incapaciting_Injury': 6,'Fatal_Injury': 3,'Unknown': 4,'Nonincapaciting_Evident_Injury': 5,'Died_Prior_to_Accident': 2,'Injured_Severity_Unknown': 7})
# Create target object and call it y
y = fatality_data.INJURY_SEVERITY
# Create X
features = ['AGE','Sex_binary','ALCOHOL_TEST_RESULT','DRUG_TEST_RESULTS_(1_of_3)','DRUG_TEST_RESULTS_(2_of_3)', 'DRUG_TEST_RESULTS_(3_of_3)']
fatality_data['Age'] = fatality_data['AGE'].fillna(0)
fatality_data['Sex_binary'] = fatality_data['Sex_binary'].fillna(0)

X = fatality_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
fatality_model = DecisionTreeRegressor(random_state=1)
# Fit Model
fatality_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = fatality_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
#STOP
#TRIED WORKING ON THIS TOO
#START
import pandas as pd

train = pd.read_csv('../input/fars_train.csv')
test = pd.read_csv('../input/fars_test.csv')

#Drop features we are not going to use
train = train.drop(['CASE_STATE', 'PERSON_TYPE', 'SEATING_POSITION', 'RESTRAINT_SYSTEM-USE', 'AIR_BAG_AVAILABILITY/DEPLOYMENT', 'EJECTION', 'EJECTION_PATH', 'EXTRICATION', 'NON_MOTORIST_LOCATION', 'POLICE_REPORTED_ALCOHOL_INVOLVEMENT', 'METHOD_ALCOHOL_DETERMINATION', 'ALCOHOL_TEST_TYPE', 'POLICE-REPORTED_DRUG_INVOLVEMENT', 'METHOD_OF_DRUG_DETERMINATION', 'DRUG_TEST_TYPE', 'DRUG_TEST_TYPE_(2_of_3)', 'DRUG_TEST_TYPE_(3_of_3)', 'HISPANIC_ORIGIN', 'TAKEN_TO_HOSPITAL', 'RELATED_FACTOR_(1)-PERSON_LEVEL', 'RELATED_FACTOR_(2)-PERSON_LEVEL', 'RELATED_FACTOR_(3)-PERSON_LEVEL', 'RACE'],axis=1)

test = test.drop(['CASE_STATE'],axis=1)
#Look at the first 3 rows of our training data
#train.head(3)
test.head(3)



#Convert ['male','female'] to [1,0] so that our decision tree can be built also convert for injury_severity too
for df in [train]:
    df['Sex_binary']=df['SEX'].map({'male':1,'female':0})
    
  
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['AGE'] = train['AGE'].fillna(0)
train['Sex_binary'] = train['Sex_binary'].fillna(0)
train['INJURY_SEVERITY'] = train['INJURY_SEVERITY'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['AGE','Sex_binary','ALCOHOL_TEST_RESULT','DRUG_TEST_RESULTS_(2_of_3)','DRUG_TEST_RESULTS_(2_of_3)', 'DRUG_TEST_RESULTS_(3_of_3)']
target = 'INJURY_SEVERITY'

#Look at the first 3 rows (we have 100 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(3)
#Display first 3 target variables
data_without_missing_values = train.dropna(axis=1)
df['INJURY_SEVERITY']=df['INJURY_SEVERITY'].map({'Possible_Injury': 0,'No_Injury': 1,'Incapaciting_Injury': 6,'Fatal_Injury': 3,'Unknown': 4,'Nonincapaciting_Evident_Injury': 5,'Died_Prior_to_Accident': 2,'Injured_Severity_Unknown': 7})
train[target].head(3).values

from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 
predictions = clf.predict(train[features])
predictions
submission = pd.DataFrame({'AGE':train['AGE'],'Sex_binary':train['Sex_binary'],'ALCOHOL_TEST_RESULT':train['ALCOHOL_TEST_RESULT'],'DRUG_TEST_RESULTS_(2_of_3)':train['DRUG_TEST_RESULTS_(2_of_3)'],'DRUG_TEST_RESULTS_(2_of_3)':train['DRUG_TEST_RESULTS_(2_of_3)'], 'DRUG_TEST_RESULTS_(3_of_3)':train['DRUG_TEST_RESULTS_(3_of_3)'],'INJURY_SEVERITY':predictions})
submission.head()
filename = 'Fatality Analysis Reporting System Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
#STOP
#FINISHED THE PREDICTION HERE