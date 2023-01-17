# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# check the data



train.head()

#test.head()
# checking the correlation between  survived and others

sns.heatmap(train.corr())
# based on the dataset, examine overall chance of the passengers surviving from train dataset.

train['Survived'].mean()



# result show on average, only 38 % of the total passenger survived
# check chances of survival for each class

train.groupby('Pclass').mean()



#based on class, 62% from Pclass 1 survived, 47% from Pclass 2, and only a mere 24% from class 3 survived 
train.groupby(['Pclass', 'Sex']).mean()



# based on class and sex, 96% of female in class 1 survived compared to only 36% men. In class 2, 92% female survive while male only 15%.

# In class 3, 50% of female survived while only a mere 13% of men survived.
# group the age by increment of 10

group_by_age = pd.cut(train['Age'], np.arange(0,90,10))

train.groupby(group_by_age).mean()



# around 59% of children survived

# around 38% of teenagers survived

# around 36% 20-30 survived

# around 44% 30-40

# around 38% 40-50

# around 40% 50-60

# around 23% 60-70

# around 20% 70-80
# just to see if sex together with age affect survived

group_by_age= pd.cut(train['Age'], np.arange(0,90,10))

train.groupby([group_by_age, 'Sex']).mean()
train.count()
test.count()
train['Age'].unique()
train.shape
# There are two genders and three passenger classes in this dataset. 

# So we create a 2 by 3 matrix to store the median values.

 

# Create a 2 by 3 matrix of zeroes

median_ages = np.zeros((2,3))

combine = [train, test]

 

# For each cell in the 2 by 3 matrix

for i in range(0,2):

    for j in range(0,3):

 

    	# Set the value of the cell to be the median of all `Age` values

    	# matching the criterion 'Corresponding gender and Pclass',

    	# leaving out all NaN values

        median_ages[i,j] = combine[ (combine['Gender'] == i) & (combine['Pclass'] == j+1)]['Age'].dropna().median()

 

# Create new column AgeFill to put values into. 

# This retains the state of the original data.

combine['AgeFill'] = combine['Age']

combine[ combine['Age'].isnull()][['Age', 'AgeFill', 'Gender', 'Pclass']].head(10)

 

# Put our estimates into NaN rows of new column AgeFill.

# df.loc is a purely label-location based indexer for selection by label.

 

for i in range(0, 2):

    for j in range(0, 3):

 

    	# Locate all cells in dataframe where `Gender` == i, `Pclass` == j+1

    	# and `Age` == null. 

    	# Replace them with the corresponding estimate from the matrix.

        combine.loc[ (combine.Age.isnull()) & (combine.Gender == i) & (combine.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
test.count()
#drop cabin from both train and test

train = train.drop('Cabin', axis=1)

test = test.drop('Cabin', axis=1)
train.count()
test.count()
train.head()

#drop passengerid

train = train.drop("PassengerId", axis = 1)

test_id = test['PassengerId']

test = test.drop("PassengerId", axis = 1)



# drop name

train = train.drop("Name", axis = 1)

test = test.drop("Name", axis = 1)



#drop ticket

train = train.drop("Ticket", axis = 1)

test = test.drop("Ticket", axis = 1)



#drop fare

train = train.drop("Fare", axis = 1)

test = test.drop("Fare", axis = 1)



#drop SibSp and Parch
#drop SibSp and Parch

train = train.drop("SibSp", axis = 1)

test = test.drop("SibSp", axis = 1)



train = train.drop("Parch", axis = 1)

test = test.drop("Parch", axis = 1)

mean_age = train['Age'].mean()



train['Age'] = train['Age'].fillna(mean_age)
train = train[train['Embarked'].notnull()]
train.count()
combine = [train, test]



# change Embarked and Sex

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
# preprocessing

from sklearn import preprocessing



le = preprocessing.LabelEncoder()

# change sex and embarked to numerical
test.head()
train.head()
from sklearn.tree import DecisionTreeClassifier
X_train = train.drop("Survived", axis = 1)

Y_train = train['Survived']


test['Age'] = test['Age'].fillna(mean_age)

test.count()
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)



Y_pred = model.predict(test)

Y_pred
Y_train
acc_decision_tree = round(model.score(X_train, Y_train) * 100, 2)

acc_decision_tree
submission = pd.DataFrame({

        "PassengerId": test_id,

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)
submission