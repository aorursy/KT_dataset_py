import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
titanic_data_filepath = '../input/titanic/train.csv'

titanic_data = pd.read_csv(titanic_data_filepath, index_col='PassengerId')

titanic_data.head()
titanic_data.shape
titanic_data.isnull().sum()
no_null_vals = titanic_data.copy()

no_null_vals.isnull().sum()

no_null_vals = no_null_vals.dropna(axis='rows')
titanic_data[titanic_data['Age'].isna()]
no_null_vals.isna().sum()
import random



to_replace_c1 = ['11-20', '61-70', '71-80']

to_replace_c2 = ['41-50', '51-60']

to_replace_c3 = ['1-10', '31-40']



to_replace_cl1 = ['A', 'B', 'C']

to_replace_cl2 = ['D', 'E']

to_replace_cl3 = ['F', 'G']



def is_nan(num):

    return (num != num)



def determineAgeGroup(row):

    age = row['Age']

    if age <= 10:

        return '1-10'

    elif age <= 20:

        return '11-20'

    elif age <= 30:

        return '21-30'

    elif age <= 40:

        return '31-40'

    elif age <= 50:

        return '41-50'

    elif age <= 60:

        return '51-60'

    elif age <= 70:

        return '61-70'

    elif age <= 80:

        return '71-80'

    elif age <= 90:

        return '81-90'

    else:

        if row['Pclass'] == 1:

            return to_replace_c1[random.randint(0, len(to_replace_c1) - 1)]

        elif row['Pclass'] == 2:

            return to_replace_c2[random.randint(0, len(to_replace_c2) - 1)]

        else:

            return to_replace_c3[random.randint(0, len(to_replace_c3) - 1)]

            



def getCabinLetter(row):

    if is_nan(row['Cabin']):

        if row['Pclass'] == 1:

            return to_replace_cl1[random.randint(0, len(to_replace_cl1) - 1)]

        elif row['Pclass'] == 2:

            return to_replace_cl2[random.randint(0, len(to_replace_cl2) - 1)]

        else:

            return to_replace_cl3[random.randint(0, len(to_replace_cl3) - 1)]

    else:

        return str(row['Cabin'])[0]



def getCabinNumber(row):

    cabinArr = str(row['Cabin']).split(' ')

    return cabinArr[0][1:]



no_null_vals['Age Group'] = no_null_vals.apply(lambda row: determineAgeGroup(row), axis=1)

no_null_vals['Cabin Letter'] = no_null_vals.apply(lambda row: getCabinLetter(row), axis=1)

no_null_vals['Cabin Number'] = no_null_vals.apply(lambda row: getCabinNumber(row), axis=1)

age_group_sequence = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

mapping = {ageGroup: el for el, ageGroup in enumerate(age_group_sequence)}

key = no_null_vals['Age Group'].map(mapping)
sorted_by_age = no_null_vals.iloc[key.argsort()].copy()

sorted_by_age.head()
no_null_vals
plt.figure(figsize=(30,10))

sns.barplot(data=sorted_by_age, x='Age Group', y='Survived')
from sklearn import preprocessing



encoder = preprocessing.LabelEncoder()

no_null_vals['Encode Cabin Letter'] = encoder.fit_transform(no_null_vals['Cabin Letter'])

no_null_vals
cabin_sequence = 'A B C D E F G T'.split(' ')

print(cabin_sequence)

cabin_mapping = {cabinGroup: el for el, cabinGroup in enumerate(cabin_sequence)}

cabin_key = no_null_vals['Cabin Letter'].map(cabin_mapping)



sorted_by_cabin = no_null_vals.iloc[cabin_key.argsort()]



plt.figure(figsize=(30,10))

sns.barplot(data=sorted_by_cabin, x='Cabin Letter', y='Survived')
plt.figure(figsize=(30,10))

sns.barplot(data=sorted_by_age, x='Age Group', y='Pclass')
plt.figure(figsize=(30,10))

sns.barplot(data=sorted_by_cabin, x='Cabin Letter', y='Pclass')
plt.figure(figsize=(30,10))

sns.scatterplot(data=no_null_vals, x='Fare', y='Encode Cabin Letter')


filled_data = titanic_data.copy()





filled_data['Age Group'] = filled_data.apply(lambda row: determineAgeGroup(row), axis=1)

filled_data['Cabin Letter'] = filled_data.apply(lambda row: getCabinLetter(row), axis=1)

#filled_data['Cabin Number'] = filled_data.apply(lambda row: getCabinNumber(row), axis=1)



filled_data.head()
no_null_vals.head()
sns.set(style="white")



correlation_data = no_null_vals.corr()



mask = np.triu(np.ones_like(correlation_data, dtype=np.bool))

f, ax = plt.subplots(figsize=(30, 20))

sns.heatmap(correlation_data, mask=mask, annot=True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

columns_X = ['Age', 'Encode Cabin Letter', 'Fare', 'Pclass']

columns_y = ['Survived']

X = no_null_vals[columns_X]

y = no_null_vals[columns_y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)



model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

new_df = y_test.copy()

new_df['Predictions'] = predictions

new_df
def checkIfCorrect(row):

    if row['Survived'] == row['Predictions']:

        return True

    return False

new_df['isCorrect'] = new_df.apply(lambda row: checkIfCorrect(row), axis=1)
new_df['isCorrect'].value_counts()
filled_data.head()
encoder = preprocessing.LabelEncoder()

filled_data['Encode Sex'] = encoder.fit_transform(filled_data['Sex'])

filled_data.head()
filled_data['Encode Cabin Letter'] = encoder.fit_transform(filled_data['Cabin Letter'])

filled_data['Encode Age Group'] = encoder.fit_transform(filled_data['Age Group'])
filled_data.head()
X_columns = [

    'Pclass',

    'Encode Sex',

    'Encode Age Group',

    'Fare',

    'Encode Cabin Letter'

]



y_columns = ['Survived']



X = filled_data[X_columns]

y = filled_data[y_columns]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
model = LogisticRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)



check_predictions = y_test.copy()

check_predictions['Predictions'] = predictions
check_predictions
check_predictions['isCorrect'] = check_predictions.apply(lambda row: checkIfCorrect(row), axis=1)
from sklearn.metrics import accuracy_score

check_predictions['isCorrect'].value_counts()

accuracy_score(predictions, y_test)