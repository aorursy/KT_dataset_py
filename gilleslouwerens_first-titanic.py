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
# Import libraries in python

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import random

from ggplot import *

import seaborn as sns

sns.set_style('whitegrid')



# Plot matplotlib inline

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
## Print python versions installed:

print('Python version ' + sys.version)

print('Pandas version: ' + pd.__version__)

print('Matplotlib version ' + matplotlib.__version__)
## Loading data (train.csv and test.csv) and changing index to 'PassengerID'



Location_train = r'../input/train.csv'

Location_test = r'../input/test.csv'

df_train = pd.read_csv(Location_train)

df_train = pd.DataFrame(df_train, index=[df_train['PassengerId']])

df_test = pd.read_csv(Location_test)

df_test = pd.DataFrame(df_test)



print('amount of samples in training data: ' + str(len(df_train)))

print('amount of samples in test data: ' + str(len(df_test)))
print(df_train.describe())

print('-------------------------------------')

print(df_test.describe())
df_train.info()

print('-------------------------------------')

df_test.info()
# Because the column 'Cabin' has a lot of nan values, we drop the column:

df_train = df_train.drop('Cabin', axis=1)

df_test = df_test.drop('Cabin', axis=1)



# The 891th row of the training set seems to be a NaN row. this one is dropped:

df_train = df_train.drop(df_train.index[890])



# PassengerID is used as index, and does not add anything to the analytics. therefore this column will also be 

# dropped

df_train = df_train.drop('PassengerId', axis=1)

df_test = df_test.drop('PassengerId', axis=1)



# Now Ticket column will also be dropped:

df_train = df_train.drop('Ticket', axis=1)

df_test = df_test.drop('Ticket', axis=1)
###################  AGE  #########################



# Only 177 of 890 samples in the trainingset do not have an age. This will now be filled by randomly inserted

# gaussian distributed data.



# Plot the age column

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values')

axis2.set_title('New Age values')

axis1.set_xlabel('Age')

axis2.set_xlabel('Age')



df_train['Age'].plot.hist(40, ax=axis1)

plt.grid(True)





# Find mean and std of complete dataset (train and test)

df_age_mean = np.mean([df_train['Age'].mean(), df_test['Age'].mean()])

df_age_std = np.mean([df_train['Age'].std(), df_test['Age'].std()])

# Find indexes of NaN values in the age column:

train_nan_age_index =  df_train['Age'].index[df_train['Age'].apply(np.isnan)]

test_nan_age_index =  df_test['Age'].index[df_test['Age'].apply(np.isnan)]



# Generate random values between mean - std and mean + std:

rand_train = np.random.randint(df_age_mean - df_age_std, df_age_mean + df_age_std, size = len(train_nan_age_index))

#rand_train = pd.DataFrame(data = rand_train, index = train_nan_age_index, columns = ['Age'])



rand_test = np.random.randint(df_age_mean - df_age_std, df_age_mean + df_age_std, size = len(test_nan_age_index))

#rand_test = pd.DataFrame(data = rand_test, index = test_nan_age_index, columns = ['Age'])



# Replace NaN values with randomly generated values between mean - std and mean + std:

df_train['Age'][np.isnan(df_train['Age'])] = rand_train

df_test['Age'][np.isnan(df_test['Age'])] = rand_test



df_train['Age'].plot.hist(40, ax=axis2)

plt.grid(True)



# Replace 'Embarked' column by numeric (Int) classification:



# Embarked location: S = 0, C = 1, Q = 2:

## df['Embarked'] = df['Embarked'].fillna(0)

df_train['Embarked'].loc[df_train['Embarked'] == 'S'] = 0

df_train['Embarked'].loc[df_train['Embarked'] == 'C'] = 1

df_train['Embarked'].loc[df_train['Embarked'] == 'Q'] = 2



df_test['Embarked'].loc[df_test['Embarked'] == 'S'] = 0

df_test['Embarked'].loc[df_test['Embarked'] == 'C'] = 1

df_test['Embarked'].loc[df_test['Embarked'] == 'Q'] = 2



# Plotting the frequency of embarked passengers per port:

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Training set')

axis2.set_title('Test set')

df_train['Embarked'].plot.hist(ax=axis1)

df_test['Embarked'].plot.hist(ax=axis2)



# Plotting the Embarked column vs. the fare rate. This obviously shows that passengers embarked in 1 (C, or Cherbourg)

# paid more for their Fare, but I will not use this for now. 

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,15))

axis1.set_title('Training set')

axis2.set_title('Test set')

sns.boxplot(x="Embarked", y="Fare", data=df_train, whis=np.inf, ax=axis1)

sns.stripplot(x="Embarked", y="Fare", data=df_train, jitter=True, color=".3", ax=axis1)



sns.boxplot(x="Embarked", y="Fare", data=df_test, whis=np.inf, ax=axis2)

sns.stripplot(x="Embarked", y="Fare", data=df_test, jitter=True, color=".3", ax=axis2)
# one more nice plot using Seaborn:



sns.factorplot('Embarked','Survived', data=df_train,size=4,aspect=3)

plt.show()



sns.boxplot('Survived','Age', data=df_train)

sns.stripplot('Survived','Age', data=df_train)
# Because by far most of the passengers embarked in 0 (S, Southampton) the NaN values will be assigned to this port:

df_train['Embarked'] = df_train['Embarked'].fillna(0)

df_test['Embarked'] = df_test['Embarked'].fillna(0)
# Now we will change the Sex column to an integer column, meaning Female will become 1 and Male will be 0:

df_train['Sex'].loc[df_train['Sex'] == 'male'] = int(0)

df_train['Sex'].loc[df_train['Sex'] == 'female'] = int(1)



df_test['Sex'].loc[df_test['Sex'] == 'male'] = int(0)

df_test['Sex'].loc[df_test['Sex'] == 'female'] = int(1)

# the Fare column in the test set seems to be missing a value. we will replace this by the mean of all fares:

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
# create a new column that incorporates whether or not the passenger had a title:



import re



# A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase 

    # letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# Get all the titles and print how often each one occurs.

titles = df_train["Name"].apply(get_title)

print(pd.value_counts(titles))



# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, \

                 "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, \

                 "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



# Verify that we converted everything.

print(pd.value_counts(titles))



# Add in the title column.

df_train["Title"] = titles



# Apply the same to the test set:

titles_test = df_test["Name"].apply(get_title)



# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, \

                 "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, \

                 "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 2}

for k,v in title_mapping.items():

    titles_test[titles_test == k] = v

    

df_test["Title"] = titles_test
print('Training dataframe status:')

print(df_train.info())

print('______________________________________________________________________________')

print('Test dataframe status:')

print(df_test.info())



df_test.describe()
# Making sure both dataframes have the same and correct datatype:

df_train['SibSp'] = df_train['SibSp'].astype(int)

df_train['Parch'] = df_train['Parch'].astype(int)

df_train['Pclass'] = df_train['Pclass'].astype(int)

df_train['Sex'] = df_train['Sex'].astype(int)

df_test['Sex'] = df_test['Sex'].astype(int)

df_train['Title'] = df_train['Title'].astype(int)

df_test['Title'] = df_test['Title'].astype(int)
# Convert 'SibSp' and 'Parch' into one column; family. then convert to 1 if > 0.

df_train['Family'] = df_train['SibSp'].copy() + df_train['Parch'].copy()

df_train['Family'].loc[df_train['Family'] > 3] = 1



df_test['Family'] = df_test['SibSp'].copy() + df_test['Parch'].copy()

df_test['Family'].loc[df_test['Family'] > 3] = 1



# Create a new column that incorporates whether or not the passenger is a Child (age < 12):



df_train['Child'] = df_train['Sex']

df_train['Child'].loc[df_train['Age'] <= 12] = 1



df_test['Child'] = df_test['Sex']

df_test['Child'].loc[df_test['Age'] <= 12] = 1
# Now we will calculate the covariance for each feature compared to our y-value ('Survived')



from sklearn.feature_selection import SelectKBest, f_classif



# Feature selection

predictors = ['Pclass', 'Sex', 'Title', 'Fare', 'Family']

selector = SelectKBest(f_classif, k=5)

selector.fit(df_train[predictors], df_train['Survived'])



# derive p-values:

p_values = selector.pvalues_

# transform p-values into scores:

scores = -np.log10(p_values)



# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()

# define training and testing sets







X_train = df_train[predictors]

Y_train = df_train["Survived"]

X_test  = df_test[predictors]
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=50)



kf = random_forest



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
# Because my first submission showed a big decrease in efficiency (only about 72%), I want to reduce the

# overfitting by doing cross-validation.



rand_forest_kf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=2, min_samples_leaf=1)



kf = cross_validation.KFold(df_train[predictors].shape[0], n_folds=3, random_state=1)



scores = cross_validation.cross_val_score(rand_forest_kf, X_train, Y_train, cv=kf)



print(scores)
# Create a new dataframe with only the columns Kaggle wants from the dataset.

df_deliverable = pd.read_csv(Location_test)

df_deliverable = pd.DataFrame(df_deliverable)



Y_pred = Y_pred.astype(int)



submission = pd.DataFrame({

        "PassengerId": df_deliverable["PassengerId"],

        "Survived": Y_pred })

submission.to_csv('first_titanic_submission.csv', index=False)