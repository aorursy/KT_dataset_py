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
datapath = "../input/"
train_file = pd.read_csv(datapath+"train.csv")
test_file = pd.read_csv(datapath+"test.csv")
print(train_file.head(2))
print(test_file.head(2))
#Removing Names and Subtypes of Outcome, for now
train_file.drop(["Name", "OutcomeSubtype"], axis=1, inplace=True)
test_file.drop(["Name"], axis=1, inplace=True)
train_file.head()
test_file.head()
#Converting Dates to categorical Year, Month and Day of the Week

from datetime import datetime
def convert_date(dt):
    d = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    return d.year, d.month, d.isoweekday()

train_file["Year"], train_file["Month"], train_file["WeekDay"] = zip(*train_file["DateTime"].map(convert_date))
test_file["Year"], test_file["Month"], test_file["WeekDay"] = zip(*test_file["DateTime"].map(convert_date))

train_file.drop(["DateTime"], axis=1, inplace=True)
test_file.drop(["DateTime"], axis=1, inplace=True)
#Separating IDs
train_id = train_file[["AnimalID"]]
test_id = test_file[["ID"]]
train_file.drop(["AnimalID"], axis=1, inplace=True)
test_file.drop(["ID"], axis=1, inplace=True)
#Target variable
train_outcome = train_file["OutcomeType"]
train_file.drop(["OutcomeType"], axis=1, inplace=True)
#Converting Age to months
def age_to_months(age1):
    if age1 is np.nan:
        return 11.0
    parts = age1.split()
    if parts[0] == '0':
        return 6.0
    if parts[1] == "weeks":
        return float(parts[0]) * 0.25
    elif parts[1] == "months":
        return float(parts[0])
    else:
        return float(parts[0]) * 12
train_file.head()
train_file["AgeuponOutcome"] = train_file["AgeuponOutcome"].map(age_to_months)
test_file["AgeuponOutcome"] = test_file["AgeuponOutcome"].map(age_to_months)
train_file.head()
#Checking that train and test sets are similar
print(train_file.head(1))
print(test_file.head(1))
categorical_variables = ['AnimalType', 'SexuponOutcome', 'Breed', 'Color', 'Year', 'Month', 'WeekDay']
#Mark the training set
train_file["Train"] = 1
test_file["Train"] = 0

#Concatenate the sets
conjunto = pd.concat([train_file, test_file])
conjunto.head()
#Get the encoded set
conjunto_encoded = pd.get_dummies(conjunto, columns=categorical_variables)
conjunto_encoded.head()
#Separate the sets
train = conjunto_encoded[conjunto_encoded["Train"] == 1]
test = conjunto_encoded[conjunto_encoded["Train"] == 0]
train = train.drop(["Train"], axis=1)
test = test.drop(["Train"], axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, train_outcome, test_size=0.15)
train_outcome.head()
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=500, n_jobs=4)
forest.fit(X_train, y_train)
y_pred_val = forest.predict(X_val)
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_val, y_pred_val))
print("Accuracy: %1.3f" % accuracy_score(y_val, y_pred_val))
#Retraining with the complete training set
forest.fit(train, train_outcome)
#Getting predicted probabilities
y_pred = forest.predict_proba(test)
results = pd.read_csv(datapath+"sample_submission.csv")
results['Adoption'], results['Died'], results['Euthanasia'], results['Return_to_owner'], results['Transfer'] = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3], y_pred[:,4]
#Submission File
results.to_csv("simple_RF_submission.csv", index=False)
