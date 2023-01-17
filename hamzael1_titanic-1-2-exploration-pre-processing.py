

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

IN_CLOUD  = True

INPUT_DIR = '../input' if IN_CLOUD else './data'
train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
# Overview of the Dataset



print(train_df.dtypes)

train_df.sample(8)

# Check Missing Values



print(f'Count of Missing Values for each Column (out of {len(train_df)}): ')

print(train_df.isnull().sum())

train_df.drop('Cabin', axis=1, inplace=True)
train_df['Embarked'].value_counts()


train_df.Embarked.fillna('S', inplace=True)

assert not train_df.Embarked.isnull().any()

missing_age_rows = train_df[train_df['Age'].isnull()].copy() # Save for later processing

missing_age_rows.sample(3)
train_df.Age.fillna(train_df['Age'].mean(), inplace=True)

assert not train_df.Age.isnull().any()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
COLOR_SURVIVED='#57e8fc'

COLOR_DEAD='#fc5e57'
# Survival Ratio



labels = ['Dead', 'Survived']

val_counts = train_df.Survived.value_counts()

#print(vals)

sizes = [val_counts[0], val_counts[1]]

colors = [COLOR_DEAD, COLOR_SURVIVED ]

#print(sizes)



fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, shadow=True, startangle=90, explode=(0.1,0), autopct='%1.1f%%', colors=colors)

ax.axis('equal')

plt.title('Overall Survival Ratio')

plt.show()

def encode_sex(sex_col):

    return sex_col.map({'female': 0, 'male': 1}).astype('int')
train_df.Sex = encode_sex(train_df.Sex)

print(train_df.Sex.dtype)

train_df.Sex.unique()
COLOR_MALE   = '#6699ff'

COLOR_FEMALE = '#ff66ff'
val_counts = train_df.Sex.value_counts()

sizes  = [val_counts[0], val_counts[1]]

labels = ['Female', 'Male']

colors = [COLOR_FEMALE, COLOR_MALE]



print(val_counts, labels)

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, shadow=True, startangle=90, explode=(0.1, 0), autopct='%1.1f%%', colors= colors)

ax.axis('equal')

plt.title('Count of Passengers by Sex')

plt.show()
ct = pd.crosstab(train_df.Sex, train_df.Survived)



ind = np.arange(2)

survived_vals = [ct.loc[1][1], ct.loc[0][1]]

dead_vals = [ct.loc[1][0], ct.loc[0][0]]

print(ct)



width=0.3



plt.bar(ind, survived_vals, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, dead_vals, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, ('Men', 'Women'))

plt.yticks(np.arange(0, 600, 50))

plt.legend( loc='upper right')

plt.show()



#ax.bar(ct)
def construct_age_cat_col(age_col):

    age_cat_col = pd.Series([-1] * len(age_col))

    for i, val in age_col.iteritems():

        if val < 14:                 # Kids

            age_cat_col[i] = 0

        elif val >= 14 and val < 22: # Teens

            age_cat_col[i] = 1

        elif val >= 22 and val < 35: # Adults

            age_cat_col[i] = 2

        elif val >= 35 and val < 50: # Big Adults

            age_cat_col[i] = 3

        elif val >= 50:              # Seniors

            age_cat_col[i] = 4

        else:

            raise ValueError('Preprocessing Age: Age Value unsupported ! ', val)

    return age_cat_col
print('Information about the ages of the passengers:')

#print(train_df.Age.describe())



train_df['AgeCat'] = construct_age_cat_col(train_df.Age)



train_df.sample(5)
labels = ['Kids', 'Teens', 'Adults', 'Big Adults', 'Seniors']



ct = pd.crosstab(train_df.AgeCat, train_df.Survived, margins=True)

cats = list(ct.index.values)

cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)

cats.sort()

print(cats)

sizes = list(ct.loc[cats, 'All'])

print(sizes)

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

ax.axis('equal')

plt.title('Count of Passengers by Age Category')

plt.show()
ind = np.arange(5)

width = 0.25



survivants_values = list(ct.loc[cats, 1])

deads_values = list(ct.loc[cats, 0])





plt.bar(ind,  survivants_values, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, ('Kids', 'Teens', 'Adults', 'Big Adults','Seniors'))

plt.yticks(np.arange(0, 300, 25))

plt.legend(loc='upper right')

plt.show()
ct = pd.crosstab(train_df.Pclass, train_df.Survived, margins=True)

cats = list(ct.index.values)

cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)

cats.sort()

print(cats)

sizes = list(ct.loc[cats, 'All'])

print(sizes)

labels = ['Class 1', 'Class 2', 'Class 3']

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

ax.axis('equal')

plt.title('Count of Passengers by Class')

plt.show()


survivants_values = list(ct.loc[cats, 1])

deads_values = list(ct.loc[cats, 0])



ind = np.arange(3)

width = 0.2

plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, ('1', '2', '3') )

plt.yticks(np.arange(0, 500, 50))

plt.legend(loc='upper right')

plt.show()
ct = pd.crosstab(train_df.Embarked, train_df.Survived, margins=True)

cats = list(ct.index.values)

cats.remove('All') # Remove the 'All' row which contains the total (the Margin that we added in crosstab)

print(cats)

sizes = list(ct.loc[cats, 'All'])

labels=cats

print(sizes)

fig, ax = plt.subplots()

ax.pie(sizes, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

ax.axis('equal')

plt.title('Count of passengers by Embarked')

plt.show()


survivants_values = list(ct.loc[cats, 1])

deads_values = list(ct.loc[cats, 0])



ind = np.arange(len(cats))

width = 0.2

plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, (cats) )

plt.yticks(np.arange(0, 500, 50))

plt.legend(loc='upper right')

plt.show()
def construct_nbr_relatives_col(sibsp_col, parch_col):

    return sibsp_col+parch_col
train_df['NbrRelatives'] = construct_nbr_relatives_col(train_df['SibSp'], train_df['Parch'])

train_df.sample(3)
ct = pd.crosstab(train_df.NbrRelatives, train_df.Survived)



cats = list(ct.index.values)

print(cats)



survivants_vals = ct.loc[:, 1]

deads_vals = ct.loc[:, 0]
ind = np.arange(len(cats))

width = 0.2

plt.bar(ind, survivants_vals, width, label='Survived' , color=COLOR_SURVIVED)

plt.bar(ind+width, deads_vals, width, label='Dead', color=COLOR_DEAD)

plt.xticks(ind+width/2, cats)

plt.legend(loc='upper right')

plt.show()
train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
def construct_is_alone_col(nbr_relatives_col):

    return nbr_relatives_col.apply(lambda x: True if x == 0 else False)
train_df['IsAlone'] = construct_is_alone_col(train_df.NbrRelatives)

train_df.sample(3)
ct = pd.crosstab(train_df.IsAlone, train_df.Survived)

cats = ['Alone' if x is True else 'Not Alone' for x in list(ct.index.values)]



survivants_values = list(ct.loc[:, 1])

deads_values = list(ct.loc[:, 0])



ind = np.arange(len(cats))

width = 0.3

plt.bar(ind, survivants_values, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, deads_values, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, (cats) )

plt.yticks(np.arange(0, 500, 50))

plt.legend(loc='upper right')

plt.show()
train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]*)\.', expand=False)

print('Counts of different Titles:')

train_df['Title'].value_counts()
rare_titles = ['Jonkheer', 'Don', 'Sir', 'Countess', 'Capt', 'Jonkheer', 'Dona', 'Major', 'Dr', 'Rev', 'Col', 'Lady']

train_df['Title'].replace(rare_titles, 'Rare', inplace=True)

train_df[train_df['Title'] == 'Rare'].sample(3)
train_df['Title'].replace(['Ms', 'Mme', 'Mlle', 'Mrs'], 'Miss', inplace=True)
def construct_title_col(name_col):

    title_col = name_col.str.extract(r'([A-Za-z]*)\.', expand=False)

    rare_titles = ['Jonkheer', 'Don', 'Sir', 'Countess', 'Capt', 'Jonkheer', 'Dona', 'Major', 'Dr', 'Rev', 'Col', 'Lady']

    title_col.replace(rare_titles, 'Rare', inplace=True)

    title_col.replace(['Ms', 'Mme', 'Mlle', 'Mrs'], 'Miss', inplace=True)

    return title_col
# Check if working by droping the Column and creating it again using the function

train_df.drop('Title', axis=1, inplace=True)

train_df['Title'] = construct_title_col(train_df['Name'])

train_df.sample(5)
train_df[train_df['Title']=='Master'].sample(5)
train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId'])) , 'Age'] = np.NaN

mean_age_masters = train_df.loc[ (train_df['Title']=='Master') ].Age.mean()

print('Mean of Master passengers\'s ages :' , mean_age_masters)

print('Number of Master passengers who will be affected by the change: ', train_df[train_df['Title'] == 'Master']['Age'].isnull().sum())
train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId'])) , 'Age'] = mean_age_masters

train_df.loc[ (train_df['Title']=='Master') & (train_df['PassengerId'].isin(missing_age_rows['PassengerId']))]
# Check

train_df[train_df['Title'] == 'Master']['Age'].max()
ct = pd.crosstab(train_df.Title, train_df.Survived)

cats = list(ct.index.values)

print(cats)

#ct


survivants_vals = ct.loc[cats, 1]

deads_vals = ct.loc[cats, 0]



ind = np.arange(len(cats))

width = 0.25

plt.bar(ind, survivants_vals, width, label='Survived', color=COLOR_SURVIVED)

plt.bar(ind+width, deads_vals, width, label='Dead', color=COLOR_DEAD)



plt.xticks(ind+width/2, cats)

plt.legend(loc='upper right')

plt.show()
train_df.drop('Name', axis=1, inplace=True)
test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')

test_df.sample(5)
print(f'Checking for Missing values in Test Dataset (out of {len(test_df)}): ')

test_df.isnull().sum()
test_df['Title'] = construct_title_col(test_df['Name'])

test_df.sample(3)
test_df['NbrRelatives'] = construct_nbr_relatives_col(test_df['SibSp'], test_df['Parch'])

test_df.sample(3)
test_df['IsAlone'] = construct_is_alone_col(test_df['NbrRelatives'])

test_df.sample(3)
print('Nbr of Missing Age Values for passengers with Master Title', test_df[test_df['Title'] == 'Master'].Age.isnull().sum())
mean_ages_masters = test_df[test_df['Title'] == 'Master'].Age.mean()

print(mean_ages_masters)

test_df.loc[ (test_df['Title'] == 'Master') & (test_df['Age'].isnull()), 'Age'] = mean_ages_masters



assert not test_df[test_df['Title'] == 'Master'].Age.isnull().any()
print('Nbr of Missing Age Values for passengers except Master Title', test_df[test_df['Title'] != 'Master'].Age.isnull().sum())
mean_ages_all = test_df.Age.mean()

print(mean_ages_all)

test_df['Age'] = test_df.Age.fillna( mean_ages_all )



assert not test_df.Age.isnull().any()
test_df.loc[test_df.Fare.isnull(), :]
similar_passengers = test_df.loc[(test_df.Pclass == 3) & (test_df.IsAlone == True) & (test_df.Embarked == 'S'), :]

similar_passengers.sample(3)
assert test_df.Fare.isnull().any()

similar_passengers_mean_fare = similar_passengers.Fare.mean()

print('Mean Fare of Similar Passengers: ', similar_passengers_mean_fare)

test_df.Fare.fillna(similar_passengers_mean_fare, inplace=True)

assert not test_df.Fare.isnull().any()

# Check the Passenger

test_df.loc[test_df.PassengerId == 1044,:]
test_df['AgeCat'] = construct_age_cat_col(test_df['Age'])

test_df.sample(3)
def encode_embarked(embarked_col):

    #return embarked_col.map({'S': 2, 'Q': 1, 'C': 0}).astype('int')

    return pd.get_dummies(data=embarked_col, columns=['Embarked'], prefix='Embarked')
one_hot_embarked_cols = encode_embarked(train_df['Embarked'])

train_df = pd.concat([train_df, one_hot_embarked_cols], axis=1)

train_df.sample(3)
one_hot_embarked_cols = encode_embarked(test_df['Embarked'])

test_df = pd.concat([test_df, one_hot_embarked_cols], axis=1)

test_df.sample(3)
train_df.drop('Embarked', axis=1, inplace=True)

test_df.drop('Embarked', axis=1, inplace=True)
def encode_title(title_col):

    #return title_col.map({ 'Mr': 0, 'Miss': 1, 'Master': 2, 'Rare': 3 }).astype('int')

    return pd.get_dummies(data=title_col, prefix='Title')
one_hot_title_cols = encode_title(train_df['Title'])

train_df = pd.concat([train_df, one_hot_title_cols], axis=1)
one_hot_title_cols = encode_title(test_df['Title'])

test_df = pd.concat([test_df, one_hot_title_cols], axis=1)
train_df.sample(3)
test_df.sample(3)
train_df.drop('Title', axis=1, inplace=True)

test_df.drop('Title', axis=1, inplace=True)
test_df.Sex = encode_sex(test_df.Sex)

test_df.sample(3)
train_df.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
print('Check columns of both datasets:')

train_df.columns, test_df.columns
OUTPUT = True

OUTPUT_DIR = '.' if IN_CLOUD else INPUT_DIR

if OUTPUT:

    train_df.to_csv(f'{OUTPUT_DIR}/train_clean.csv', index=False)

    test_df.to_csv(f'{OUTPUT_DIR}/test_clean.csv', index=False)

    print('Done Outputing to CSV')

train_df.sample(5)
test_df.sample(5)