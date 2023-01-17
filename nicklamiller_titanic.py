# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

display(train_data.head())

display(test_data.head())

display(train_data.info());
display(train_data.describe())

display(test_data.describe())
# find number and percentage of NA's

display(train_data.isna().sum())

display(train_data.isna().mean()*100)
# See how missing values correlate to ticket class

plt.figure(figsize=(40,10))

plt.xticks(fontsize=30), plt.yticks(fontsize=20)

plt.title('Counts of missing Age values for different Pclass',fontsize=40)

train_data.loc[train_data['Age'].isna()]['Pclass'].value_counts().plot(kind='bar');

plt.figure(figsize=(40,10))

plt.xticks(fontsize=30), plt.yticks(fontsize=20)

plt.title('Counts of missing Cabin values for different Pclass',fontsize=40)

train_data.loc[train_data['Cabin'].isna()]['Pclass'].value_counts().plot(kind='bar');
# See if age makes a substantial difference in survival rate

plt.figure(figsize=(40,10))

label_size = 30

plt.xlabel('Age',size = label_size), plt.ylabel('Frequency',size = label_size)

plt.xticks(fontsize=30), plt.yticks(fontsize=20)

plt.title("Histogram of Age for those who survived and those who didn't",fontsize=40)

no_bins = 15

alpha_level = 0.5

train_data.loc[train_data['Survived']==1]['Age'].plot(kind='hist',

                                                      bins=no_bins,

                                                      alpha = alpha_level,

                                                      density=True,

                                                      label='Survived')

train_data.loc[train_data['Survived']==0]['Age'].plot(kind='hist',

                                                      bins=no_bins,

                                                      alpha = alpha_level,

                                                      density=True,

                                                      label='Not Survived')

plt.legend(loc='upper right',prop={'size':30});
# explore sex since it is known to be a fairly impactful factor in determining survival

women = train_data.loc[train_data.Sex=='female']["Survived"]

rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex=='male']["Survived"]

rate_men = sum(men)/len(men)



print("% of women who survived:", rate_women)

print("% of men who survived:", rate_men)
# Get rid of Cabin column

train_data.drop(['Cabin'], axis=1,inplace=True)

train_data.head()
# Get rid of Cabin column

test_data.drop(['Cabin'], axis=1,inplace=True)

test_data.head()
# generate pairwise plots between the numeric attributes

sns.pairplot(train_data._get_numeric_data());
# generate heatmap of the attributes' correlations

corr = train_data.corr()

plt.figure(figsize=(30,10))

sns.heatmap(corr,annot=True);
# KNN implementation

from sklearn.impute import KNNImputer



imputer = KNNImputer(n_neighbors=3)

train_data_filled = imputer.fit_transform(train_data._get_numeric_data())
# see how imputed values compare to original values, view tail since NA's in age

display(train_data._get_numeric_data().tail(5))

display(train_data_filled[-6:len(train_data_filled)])

display(train_data_filled[-6:len(train_data_filled),2:7]) # view middle columns
# convert filled data into a dataframe

train_data_filled = pd.DataFrame(data=train_data_filled,

                                 columns=["PassengerId", "Survived","Pclass","Age","SibSp","Parch","Fare"])
train_data['Embarked'].fillna(train_data['Embarked'].value_counts()

.idxmax(), inplace=True)
# convert numpy matrix into pandas dataframe

cols_to_use = train_data.columns.difference(train_data_filled.columns)

train_data = pd.merge(train_data_filled,train_data[cols_to_use], left_index=True,right_index=True)

train_data.isna().sum()
# KNN implementation

from sklearn.impute import KNNImputer



imputer = KNNImputer(n_neighbors=3)

test_data_filled = imputer.fit_transform(test_data._get_numeric_data())
# see how imputed values compare to original values, view tail since NA's in age

display(test_data._get_numeric_data().tail(5))

display(test_data_filled[-6:len(test_data_filled)])

display(test_data_filled[-6:len(test_data_filled),2:7]) # view middle columns
# convert filled data into a dataframe

test_data_filled = pd.DataFrame(data=test_data_filled,

                                 columns=["PassengerId", "Pclass","Age","SibSp","Parch","Fare"])
# convert numpy matrix into pandas dataframe

cols_to_use = test_data.columns.difference(test_data_filled.columns)

test_data = pd.merge(test_data_filled,test_data[cols_to_use], left_index=True,right_index=True)

test_data.isna().sum()
# replace categorical data with dummy variables using label encoding for Sex

# and one hot encoding for Embarked

sex = pd.get_dummies(train_data['Sex'], drop_first=True)

embark = pd.get_dummies(train_data['Embarked'], drop_first=True)

train_data = pd.concat([train_data, sex, embark], axis=1)

train_data.head()
# replace categorical data with dummy variables using label encoding for Sex for test_data

# and one hot encoding for Embarked

sex = pd.get_dummies(test_data['Sex'], drop_first=True)

embark = pd.get_dummies(test_data['Embarked'], drop_first=True)

test_data = pd.concat([test_data, sex, embark], axis=1)

test_data.head()
# Drop columns for which dummy variables were made, also drop name and ticket columns

test_data.drop(["Name","Sex","Embarked","Ticket"],axis=1,inplace=True)

test_data.head()
# Drop columns for which dummy variables were made, also drop name and ticket columns

train_data.drop(["Name","Sex","Embarked","Ticket"],axis=1,inplace=True)

train_data.head()
# Split the training data into X and y features

X = train_data[["Pclass","Age","SibSp","Parch","Fare","male","Q","S"]]

y = train_data["Survived"]
train_data.info()
# Split the data into training and testing datasets

from sklearn.model_selection import train_test_split



# Split data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# Import module for fitting

from sklearn.linear_model import LogisticRegression



# Create instance (i.e. object) of LogisticRegression and fit the model using training data

logmodel = LogisticRegression().fit(X_train, y_train)
# see how model performed

from sklearn.metrics import classification_report

print(classification_report(y_test, logmodel.predict(X_test)))
# Create and submit predictions

predictions = logmodel.predict(test_data[["Pclass","Age","SibSp","Parch","Fare","male","Q","S"]])

predictions.shape
predictions = predictions.astype(np.int32)
test_data["PassengerId"] = test_data["PassengerId"].astype('int32', copy=False)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")