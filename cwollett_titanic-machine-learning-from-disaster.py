# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%pylab inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as P
from sklearn.ensemble import RandomForestClassifier 

# For .read_csv, always use header=0 when you know row 0 is the header row
#df = pd.read_csv('../input/train.csv', header=0)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df.info()
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['EmbarkLocation'] = df['Embarked'].fillna('NA').map({'C': 0, 'Q': 1, 'S': 2, 'NA': 4}).astype(int)
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()
df['AgeFill'] = df['Age']
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df['AgeFill'] * df['Pclass']
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 
train_data = df.values
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])
test_df = pd.read_csv('../input/test.csv', header=0)
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_df['EmbarkLocation'] = test_df['Embarked'].fillna('NA').map({'C': 0, 'Q': 1, 'S': 2, 'NA': 4}).astype(int)
median_ages_test = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages_test[i,j] = test_df[(test_df['Gender'] == i) & (test_df['Pclass'] == j + 1)]['Age'].dropna().median()
test_df['AgeFill'] = test_df['Age']
test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)
for i in range(0,2):
    for j in range(0,3):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill'] = median_ages_test[i,j]
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['Age*Class'] = test_df['AgeFill'] * test_df['Pclass']
fare_test = np.zeros((1,3))
for i in range(0,1):
    for j in range(0,3):
        fare_test[i,j] = test_df[(test_df['Pclass'] == j + 1)]['Fare'].dropna().mean()
for i in range(0,1):
    for j in range(0,3):
        test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == j + 1), 'Fare'] = fare_test[i,j]
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
test_data = test_df.values
# Take the same decision trees and run it on the test data
output = forest.predict(test_data)
print(output)
