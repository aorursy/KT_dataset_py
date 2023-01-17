import pandas as pd

# Read in training and test set
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Get a first impression of what kind of data is in the train set
train.head(10)
# Figure out datatypes 
train.dtypes
train.describe()
train.isnull().sum()
import matplotlib.pyplot as plt
%matplotlib inline

n, bins, patches = plt.hist(train.Age.dropna(), 20 )
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

train.Age.median()
train.Age.mean()
train.Age.mode()
train[train.Age < 5][['Age','Name']].head(20)
# The girls seems to all be Miss and the boys Master (cute!) so it seems this not a mistake. Let's leave it
train_copy = train.copy()
test_copy = test.copy()

del train_copy['PassengerId']
passengerid = test_copy['PassengerId'].copy()
test_copy = test_copy.drop(['PassengerId'], axis = 1)

del train_copy['Ticket']
del test_copy['Ticket']
train_copy.Pclass = train_copy.Pclass.astype('category')
test_copy.Pclass = test_copy.Pclass.astype('category')
train_copy.Age = train_copy.Age.fillna(train.Age.median())
test_copy.Age = test_copy.Age.fillna(train.Age.median())

# Confirm that there are no more null values in Age
train_copy.Age.isnull().sum()
train_copy.Fare = train_copy.Fare.fillna(train.Fare.median())
test_copy.Fare = test_copy.Fare.fillna(train.Fare.median())

# Confirm that there are no more null values in Age
train_copy.Fare.isnull().sum()
# Calculate the mode
embarkmode = train_copy['Embarked'].mode()

# Replace null values with mode (both in train and test) 
train_copy.loc[(train_copy['Embarked'].isnull()), 'Embarked'] = embarkmode.values[0]
test_copy.loc[(test_copy['Embarked'].isnull()), 'Embarked'] = embarkmode.values[0]

# Confirm that there are no more null values in Embarked
train_copy.Embarked.isnull().sum()

# Convert to categorical data type
train_copy.Embarked = train_copy.Embarked.astype('category')
test_copy.Embarked = test_copy.Embarked.astype('category')
import re 
regex_pat = re.compile(r'[1-9A-Za-z 0]*') # If Cabin contains any characters at all
train_copy.Cabin = train_copy.Cabin.str.replace(regex_pat, '1') # Replace these with a 1 (string to conform with data type)
test_copy.Cabin = test_copy.Cabin.str.replace(regex_pat, '1') # Do the same for the test set

train_copy.Cabin = train_copy.Cabin.fillna(0) # Replace NaN values in Cabin with 0
test_copy.Cabin = test_copy.Cabin.fillna(0) # Same again for the test set

# Now make it a categorical feature
train_copy.Cabin = train_copy.Cabin.astype('category')
test_copy.Cabin = test_copy.Cabin.astype('category')
# Function that replaces categorical variables with integers
def replace_with_int(df, column, vals, replacevals):
    for i in range(len(vals)):
        df[column] = df[column].replace(vals[i], replacevals[i])
    return df

train_copy = replace_with_int(train_copy, 'Sex', ['male','female'], [0, 1])
test_copy = replace_with_int(test_copy, 'Sex', ['male','female'], [0, 1])

train_copy = replace_with_int(train_copy, 'Embarked', ['S', 'C', 'Q'], [0, 1, 2])
test_copy = replace_with_int(test_copy, 'Embarked', ['S', 'C', 'Q'], [0, 1, 2])

# Convert categorical variables to category datatype
for col in ['Pclass', 'Sex', 'Cabin', 'Embarked']:
    train_copy[col] = train_copy[col].astype('category')
    test_copy[col] = test_copy[col].astype('category')
    
train_copy['Survived'] = train_copy['Survived'].astype('category')

train_copy.dtypes
# First isolate the title using regex
regex_pat = re.compile('(.*, )|(\\..*)')
train_copy['Title'] = train_copy.Name.str.replace(regex_pat, '')
test_copy['Title'] = test_copy.Name.str.replace(regex_pat, '')

# If they are => 18 and Parent/Child column >0 then they are likely to be parents
are_parents_index = train_copy.loc[(train_copy['Age'] >= 18) & (train_copy['Parch'] > 0)].index
train_copy['WithChild'] = 0
train_copy.loc[are_parents_index,'WithChild'] = 1
are_parents_index = test_copy.loc[(test_copy['Age'] >= 18) & (test_copy['Parch'] > 0)].index
test_copy['WithChild'] = 0
test_copy.loc[are_parents_index,'WithChild'] = 1

# Kids that had at least one parent on board
with_parents_index = train_copy.loc[(train_copy['Age'] < 18) & (train_copy['Parch'] >= 1)].index
train_copy['NumOfParents'] = 0
train_copy.loc[with_parents_index, 'NumOfParents'] = 1
with_parents_index = test_copy.loc[(test_copy['Age'] < 18) & (test_copy['Parch'] >= 1)].index
test_copy['NumOfParents'] = 0
test_copy.loc[with_parents_index, 'NumOfParents'] = 1

# If they are >= 18 and have only one Sibling/Spouse on board, they are probably accompanied by their Wife/Husband
with_spouse_index = train_copy.loc[(train_copy['Age'] >= 18) & (train_copy['SibSp'] == 1)].index
train_copy['WithSpouse'] = 0
train_copy.loc[with_spouse_index, 'WithSpouse'] = 1
with_spouse_index = test_copy.loc[(test_copy['Age'] >= 18) & (test_copy['SibSp'] == 1)].index
test_copy['WithSpouse'] = 0
test_copy.loc[with_spouse_index, 'WithSpouse'] = 1

# It seems there are a small number of high-ranking individuals (based on their title)
train_copy['Title'].value_counts()

# Let's lump special titles together since they might not all occur in the test set, 5 seems to be a good cutoff point (doctors might be a good separate group)
special_titles = train_copy['Title'].value_counts()
special_title_list = list(special_titles[special_titles <= 10].index)
train_copy['Title'].replace(special_title_list, 'Special', inplace = True)
special_titles = test_copy['Title'].value_counts()
special_title_list = list(special_titles[special_titles <= 10].index)
test_copy['Title'].replace(special_title_list, 'Special', inplace = True)

# Convert these to numerical values so they can be used as a categorical feature
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_copy['TitleCode'] = le.fit_transform(train_copy['Title'])
test_copy['TitleCode'] = le.fit_transform(test_copy['Title'])

for col in ['TitleCode', 'WithSpouse', 'NumOfParents', 'WithChild']:
    train_copy[col] = train_copy[col].astype('category')
    test_copy[col] = test_copy[col].astype('category')
del train_copy['Name']
del train_copy['Title']
del test_copy['Name']
del test_copy['Title']
train_copy.dtypes
# Create X and y
y = train_copy['Survived'].copy()
X = train_copy.drop(['Survived'], axis = 1)

# train/test(/validation) split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print(clf.feature_importances_)
clf.score(X_train, y_train)
clf.score(X_test,y_test)
clf = RandomForestClassifier(random_state=42, n_estimators = 100)
clf.fit(X, y)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv = 5, scoring='precision')

import numpy as np
np.mean(scores)
y_pred = clf.predict(test_copy)
y_survived = pd.DataFrame(data = y_pred, columns = ['Survived'])
pd.concat([passengerid,y_survived], axis = 1).to_csv('submission.csv', index = False)