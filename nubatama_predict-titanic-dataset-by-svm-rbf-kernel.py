#Import libraries

import pandas as pd

import numpy as np



# For .read_csv, always use header=0 when you know row 0 is the header row

train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)
## Sex

train_df['Gender'] = train_df['Sex'].map( {'male':1, 'female':2} ).astype(int)



## Age

# Calculate mediain of each passenger class

median_ages = np.zeros((2,3))

for i in range(0,2):

    for j in range(0,3):

        median_ages[i,j]=train_df[

                            (train_df['Gender'] == i+1) &

                            (train_df['Pclass'] == j+1)

                           ]['Age'].dropna().median()



# Copy 'Age' to new row, 'AgeFill'

train_df['AgeFill'] = train_df['Age']



# Fill nan 'Age' by median value of each Pclass

for i in range(0,2):

    for j in range(0,3):

        train_df.loc[(train_df.Age.isnull()) & (train_df.Gender==i+1) & (train_df.Pclass==j+1), 'AgeFill'] = median_ages[i,j]

train_df['AgeIsNull'] = pd.isnull(train_df.Age).astype(int)



## Embarked 

# fill nan 'Embarked' by 'S'

train_df.loc[(train_df.Embarked.isnull()), 'Embarked'] = 'S'



# Map to integer code

train_df['EmbarkedPos'] = train_df['Embarked'].map( {'C':1, 'Q':2, 'S':3}).astype(int)



## Fare 

train_df.loc[(train_df.Fare.isnull()), 'Fare'] = 0.0
# Feture engineering

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']

train_df['GenderPclass'] = train_df['Pclass'] - train_df['Gender'] + 1

train_df['Pclass'] = train_df['Pclass'] / 3



# Drop object type data

train_df.dtypes[train_df.dtypes.map(lambda x: x=='object')]

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Fare', 'Age'], axis=1)



# Get train data

whole_data = train_df.values

whole_data_y = whole_data[:,1]

whole_data_x = whole_data[:,2:]



# Preprocessring data

from sklearn import preprocessing

whole_data_x_scaled = preprocessing.scale(whole_data_x)
from sklearn import svm

from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

%matplotlib inline



# Find better 'gamma' by default C value

param_range = np.logspace(-2, 0, 20)

print(param_range)

train_scores, test_scores = validation_curve(

    svm.SVC(C=0.6), whole_data_x_scaled, whole_data_y, param_name="gamma", param_range=param_range,

    cv=10, scoring="accuracy", n_jobs=1)



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



plt.title("Validation Curve with SVM")

plt.xlabel("$\gamma$")

plt.ylabel("Score")

plt.ylim(0.6, 1.1)

lw = 2

plt.semilogx(param_range, train_scores_mean, label="Training score",

             color="darkorange", lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.2,

                 color="darkorange", lw=lw)

plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",

             color="navy", lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2,

                 color="navy", lw=lw)

plt.legend(loc="best")

plt.show()

print(test_scores_mean)
# Find better 'C' by better gamma

param_range = np.linspace(2.0, 0.1, 10)

print(param_range)

train_scores, test_scores = validation_curve(

    svm.SVC(gamma=0.112), whole_data_x_scaled, whole_data_y, param_name="C", param_range=param_range,

    cv=10, scoring="accuracy", n_jobs=1)



train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



plt.title("Validation Curve with SVM")

plt.xlabel("C")

plt.ylabel("Score")

plt.ylim(0.7, 0.9)

lw = 2

plt.semilogx(param_range, train_scores_mean, label="Training score",

             color="darkorange", lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.2,

                 color="darkorange", lw=lw)

plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",

             color="navy", lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2,

                 color="navy", lw=lw)

plt.legend(loc="best")

plt.show()
## Create estimater

clf = svm.SVC(C=0.7,gamma=0.11)



# Fit all training data

clf.fit(whole_data_x_scaled, whole_data_y)



## Cleaning test data under same rule of training data

# Sex

test_df['Gender'] = test_df['Sex'].map( {'male':1, 'female':2} ).astype(int)



# Copy 'Age' to new row, 'AgeFill'

test_df['AgeFill'] = test_df['Age']



# Fill nan 'Age' by median value of each Pclass

for i in range(0,2):

    for j in range(0,3):

        test_df.loc[(test_df.Age.isnull()) & (test_df.Gender==i+1) & (test_df.Pclass==j+1), 'AgeFill'] = median_ages[i,j]

test_df['AgeIsNull'] = pd.isnull(test_df.Age).astype(int)



# Embarked

test_df.loc[(test_df.Embarked.isnull()), 'Embarked'] = 'S'

test_df['EmbarkedPos'] = test_df['Embarked'].map( {'C':1, 'Q':2, 'S':3}).astype(int)



# Fare

test_df.loc[test_df.Fare.isnull(), 'Fare'] = 0.0



# Add 'FamilySize', SibSp * Parch

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['GenderPclass'] = test_df['Pclass'] - test_df['Gender'] + 1

test_df['Pclass'] = test_df['Pclass'] / 3



# Drop unnecessary data

test_df = test_df.drop(['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1)



# Get test data

test_data = test_df.values

test_data = test_data[:,1:]

test_data = preprocessing.scale(test_data)



# predict

result = clf.predict(test_data)

test_df['Survived'] = result

df_test_result = test_df[['PassengerId', 'Survived']] 

df_test_result.Survived = df_test_result.Survived.astype(int)

df_test_result.to_csv('predict.csv', index=False)