import numpy as np

import pandas as pd

import string

import math

import random

from sklearn import svm

from sklearn import ensemble

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import re
df = pd.read_csv('../input/train.csv')

#result = pd.read_csv('../input/gender_submission.csv')

testing = pd.read_csv('../input/test.csv')

df.head(5)
# Getting substring for title

def getSubstring(string, substrings):

    for s in substrings:

        if string.find(s) != -1:

            return s

    return np.nan



# Replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

# Replacing missing value for age

def replace_ages(x, ageGroup):

    age = x['Age']

    if math.isnan(age):

        return ageGroup[x['Pclass']][x['SibSp']]

    else:

        return age
# Add columns of title and last name

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer'] 

df['Title'] = df['Name'].map(lambda x: getSubstring(x, title_list))

df['Title'] = df.apply(replace_titles, axis = 1)

# df['LastName'] = df['Name'].map(lambda x: x.split(',')[0])
# np.unique(df['Pclass'])

# np.unique(df['SibSp'])

# # plt.plot(df[df['Pclass'] == 2 & df['SibSp'] == 2]['Age'])

# a = df['Pclass'] == 0 

# b = df['SibSp'] == 1

# c = df['Age'].isnull() == False

# d = a.tolist() and b.tolist() and c.tolist()

# plt.hist(df[d]['Age'].tolist())

# plt.show()
# Fill the missing values, take the median

ageGroup = df.groupby(['Pclass', 'SibSp'])['Age'].median()

ageGroup[3][8] = df['Age'][df['Pclass'] == 3].median()

df['Age'] = df.apply(lambda x: replace_ages(x, ageGroup), axis = 1)

df['Age']
df.isnull().any()
# Replace carbin number, sepatared it into two preditors: Cabin and Cabin number

# [Hongbo] This might not be a good idea...

pattern = '([A-Z]+)([0-9]*)'

int(re.search(pattern, 'C85').group(2))

def replaceCabinNum(name):

    pattern = '([A-Z]+)([0-9]*)'

    num = re.search(pattern, name).group(2)

    if (num == ''):

        return 0

    else:

        return(int(num))



    

cabin_null = df['Cabin'].isnull()

cabin_list = [i for i in range(0, len(df['Cabin'])) if cabin_null[i] == False]

df['Cabin_Al'] = 'Unknown'

df['Cabin_Al'][cabin_list] = df['Cabin'][cabin_list].map(lambda x: re.search(pattern, x).group(1))

df['Cabin_num'] = 0

df['Cabin_num'][cabin_list] = df['Cabin'][cabin_list].apply(replaceCabinNum)





# np.unique(df['Pclass'])

# np.unique(df['SibSp'])

# # plt.plot(df[df['Pclass'] == 2 & df['SibSp'] == 2]['Age'])

# a = df['Pclass'] == 0 

# b = df['SibSp'] == 1

# c = df['Age'].isnull() == False

# d = a.tolist() and b.tolist() and c.tolist()

max = df[df['Cabin_Al'] == 'A'].Cabin_num.max()

df[df['Cabin_Al'] == 'A'].groupby(pd.cut(df["Cabin_num"], np.arange(0, max, 10))).Survived.mean()

#df.groupby('Cabin_Al').Cabin_num.min()
# Embark only 2 missing values (You can try to replace them)

df = df[df['Embarked'].isnull() == False]

df = df.reset_index(drop=True)
# Check the relationship between family size and survival rate

df['Famiy_size'] = df['SibSp'] + df['Parch']

df['Single'] = (df['Famiy_size'] == 1).astype('int')

df.groupby(['Famiy_size'])['Survived'].mean()
# Reassign the family size

def checkFamilySize(x):

    famSize = x['Famiy_size']

    if famSize == 0:

        return 'Single'

    elif famSize == 1:

        if (x['SibSp'] == 1):

            return 'Couple'

        else:

            return 'Small'

    elif famSize <= 3:

        return 'Small'

    elif famSize <= 6:

        return 'Median'

    else:

        return 'Large'



df['Family'] = df.apply(checkFamilySize, axis = 1)
# Check relationship between fare and survival rate

df['FareGroup'] = df['Fare'].map(lambda x: int(x/10) * 10)

fare = df.groupby(['FareGroup'])['Survived'].mean()

fare

# plt.plot(fare)

# plt.show()

# [i for i in range(0, len(fare)) if fare.iloc[i] == 0.5]


def checkFare(x):

    if x <= 40:

        return 'cheap'

    elif x <= 100:

        return 'middle'

    else:

        return 'expensive'

df['FareGroup'] = df['FareGroup'].map(checkFare)
df['AgeGroup'] = df['Age'].map(lambda x: int(x/5) * 5)

age = df.groupby(['AgeGroup'])['Survived'].mean()

np.unique(df['Pclass'])

df.head(3)
# Deal with the categorical data

df['Sex_male'] = (df['Sex'] == 'male').astype('int')

df['Title_Master'] = (df['Title'] == 'Master').astype('int')

df['Title_Miss'] = (df['Title'] == 'Miss').astype('int')

df['Title_Mr'] = (df['Title'] == 'Mr').astype('int')

df['Title_Mrs'] = (df['Title'] == 'Mrs').astype('int')

df['Cabin_A'] = (df['Cabin_Al'] == 'A').astype('int')

df['Cabin_B'] = (df['Cabin_Al'] == 'B').astype('int')

df['Cabin_C'] = (df['Cabin_Al'] == 'C').astype('int')

df['Cabin_D'] = (df['Cabin_Al'] == 'D').astype('int')

df['Cabin_E'] = (df['Cabin_Al'] == 'E').astype('int')

df['Cabin_F'] = (df['Cabin_Al'] == 'F').astype('int')

df['Cabin_G'] = (df['Cabin_Al'] == 'G').astype('int')

df['Cabin_T'] = (df['Cabin_Al'] == 'T').astype('int')

df['Cabin_Unknown'] = (df['Cabin_Al'] == 'Unknown').astype('int')

df['Embarked_C'] = (df['Embarked'] == 'C').astype('int')

df['Embarked_Q'] = (df['Embarked'] == 'Q').astype('int')

df['Embarked_S'] = (df['Embarked'] == 'S').astype('int')

df['Single'] = (df['Family'] == 'Single').astype('int')

df['Couple'] = (df['Family'] == 'Couple').astype('int')

df['SmallFamily'] = (df['Family'] == 'Small').astype('int')

df['MedianFamily'] = (df['Family'] == 'Median').astype('int')

df['LargeFamily'] = (df['Family'] == 'Large').astype('int')

df['Infant'] = (df['Age'] < 1).astype('int')

df['Elder'] = (df['Age'] > 60).astype('int')

df['Fare_low'] = (df['FareGroup'] == 'cheap').astype('int')

df['Fare_median'] = (df['FareGroup'] == 'middle').astype('int')

df['Fare_high'] = (df['FareGroup'] == 'expensive').astype('int')

df['Pclass_1'] = (df['Pclass'] == 1).astype('int')

df['Pclass_2'] = (df['Pclass'] == 2).astype('int')

df['Pclass_3'] = (df['Pclass'] == 3).astype('int')
sum(df['LargeFamily'] == 1)
# drop the unnecessary columns

data = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'Title', 'Cabin_Al', 'Family', 'Famiy_size',

               'SibSp', 'Parch', 'Pclass', 'FareGroup', 'Age', 'AgeGroup', 'Survived'], axis = 1)

response = df['Survived']

data.head(3)
# perform cross validation, 5 folds

cv_id = []

for i in range(0, 5):

    cv_id.extend([i] * 177)

cv_id.extend([0] * 4) 

random.shuffle(cv_id)
# cross validation 

rate = []

for i in range(0,5):

    train_index = [index for index in range(0, len(df)) if cv_id[index] != i]

    test_index = [index for index in range(0, len(df)) if cv_id[index] == i]

    train = data.iloc[train_index]

    train_response = response.iloc[train_index]

    test = data.iloc[test_index]

    test_response = response.iloc[test_index]

#     clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-5)

    clf = ensemble.RandomForestClassifier(n_estimators = 10)

    clf.fit(train, train_response)

    pred = clf.predict(test)

    rate.append(sum(pred == test_response) / len(test_response))

np.mean(rate)

    

    



    


