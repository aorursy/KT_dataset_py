# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

import seaborn as sns

sns.set(style = "ticks", color_codes = True)

import pylab as plot

params = {'axes.labelsize':'large', 'legend.fontsize': 20, 'figure.dpi': 100, 'figure.figsize': [25, 10]}

plot.rcParams.update(params)
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

train_data = pd.read_csv('../input/train.csv')
train.describe()
train.info()
train.isna().sum()
test.describe()
test.info()
test.isna().sum()
#Check whether male or female have higher survival number

print(train_data.groupby(['Sex', 'Survived']).agg('count')['PassengerId'])

sns.barplot(x = 'Sex', y = 'Survived', data = train_data)

sns.catplot(x = 'Sex', col = 'Survived', data = train_data, kind = 'count')

#Female have higher survival number than male
#Check how age affect the survival rate

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Died', kde = False)

plt.legend();

#Passenger with less thatn 16 - 17 year old -> have higher survival rate
#check the age below 20

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1) & (train_data['Age'] < 20)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0) & (train_data['Age'] < 20)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Died', kde = False)

plt.legend();

#Proved: Passenger with less thatn 16 - 17 year old -> have higher survival rate
#Check the age between 20 and 50

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1) & (train_data['Age'] >= 20) & (train_data['Age'] <= 50)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0) & (train_data['Age'] >= 20) & (train_data['Age'] <= 50)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Died', kde = False)

plt.legend();
#Check the age above 50

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1) & (train_data['Age'] > 50)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0) & (train_data['Age'] > 50)

sns.distplot(train_data.loc[condition, 'Age'], label = 'Died', kde = False)

plt.legend();
train_data.loc[train_data['Age'] <= 16, 'AgeStatus'] = 'Teenage'

train_data.loc[train_data['Age'] > 16, 'AgeStatus'] = 'Adult'
def getIdentity(df):

    df['Identity'] = df['Name'].map(lambda name: name.split(',')[1].split('.')[0].replace(" ",""))

    df.drop(columns = ['Name'], inplace = True)

    return df
#Check whether the identity will affect the survival rate

train_data = getIdentity(df = train_data)

print(train_data.groupby(['Identity','Survived']).agg('count')['PassengerId'])

sns.barplot(x = 'Identity', y = 'Survived', data = train_data)
Identity_dict = {

    'Mr':'Local Male',

    'Mrs':'Local Female',

    'Miss':'Local Female',

    'Master':'Professional',

    'Don':'Specialist',

    'Rev':'Professional',

    'Dr':'Professional',

    'MMe':'Foreign Female',

    'Ms':'Local Female',

    'Major':'Professional',

    'Lady':'Specialist',

    'Sir':'Specialist',

    'Mlle':'Foreign Female',

    'Col':'Specialist',

    'Capt':'Officier',

    'theCountess':'Specialist',

    'Jonkheer':'Foreign Male'

}



train_data['SocialStatus'] = train_data['Identity'].map(Identity_dict)
print(train_data.groupby(['SocialStatus','Survived']).agg('count')['PassengerId'])

sns.barplot(x = 'SocialStatus', y = 'Survived', data = train_data)
condition = (train_data['SocialStatus'] == "Local Male") | (train_data['SocialStatus'] == "Local Female")

sns.countplot(x = 'SocialStatus', hue = 'Survived', data = train_data[condition])
condition = (train_data['SocialStatus'] != "Local Male") & (train_data['SocialStatus'] != "Local Female")

sns.countplot(x = 'SocialStatus', hue = 'Survived', data = train_data[condition])
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

sns.barplot(x = 'FamilySize', y = 'Survived', data = train_data)

sns.catplot(x = 'FamilySize', col = 'Survived', data = train_data, kind = 'count')
train_data.loc[train_data['FamilySize'] == 1,'FamilyStatus'] = 'Single'

train_data.loc[(train_data['FamilySize'] > 1) & (train_data['FamilySize'] < 4),'FamilyStatus'] = 'Normal'

train_data.loc[train_data['FamilySize'] > 3,'FamilyStatus'] = 'Big'
sns.barplot(x = 'FamilyStatus', y = 'Survived', data = train_data)

sns.catplot(x = 'FamilyStatus', col = 'Survived', data = train_data, kind = 'count')
#Check whether Pclass will affect the survival rate or not

print(train_data.groupby(['Pclass','Survived']).agg('count')['PassengerId'])

sns.barplot(x = 'Pclass', y = 'Survived', data = train_data)

sns.catplot(x = 'Pclass', col = 'Survived', kind = 'count', data = train_data)

#Lower Pclass have lower survival rate
condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 1)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Survived', kde = False)

condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 0)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Died', kde = False)

plt.legend();
condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 1) & (train_data['Fare'] <= 10)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Survived', kde = False)

condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 0) & (train_data['Fare'] <= 10)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Died', kde = False)

plt.legend();
condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 1) & (train_data['Fare'] > 10) & (train_data['Fare'] <= 50)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Survived', kde = False)

condition = (train_data['Fare'].notnull()) & (train_data['Survived'] == 0) & (train_data['Fare'] >= 10) & (train_data['Fare'] <= 50)

sns.distplot(train_data.loc[condition, 'Fare'], label = 'Died', kde = False)

plt.legend();
train_data.loc[train_data['Fare'] < 15, 'FareStatus'] = 'Ticket lower than $15'

train_data.loc[train_data['Fare'] >= 15, 'FareStatus'] = 'Ticket euqal/higher than $15'
print(train_data.groupby(['Embarked','Survived']).agg('count')['PassengerId'])

sns.barplot(x = 'Embarked', y = 'Survived', data = train_data)

sns.catplot(x = 'Embarked', col = 'Survived', data = train_data, kind = 'count')
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = train_data, split = True)
#Male

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1) & (train_data['Sex'] == 'male')

sns.distplot(train_data.loc[condition, 'Age'], label = 'Male - Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0) & (train_data['Sex'] == 'male')

sns.distplot(train_data.loc[condition, 'Age'], label = 'Male - Died', kde = False)

#plt.hist([train_data[(train_data['Survived'] == 1) & (train_data['Sex'] == 'male')]['Age'], train_data[(train_data['Survived'] == 0) & (train_data['Sex'] == 'male')]['Age']], stacked = True, label = ['Survived', 'Died'])

#plt.xlabel('Age')

#plt.ylabel('Number of Survived')

plt.legend();
#Female

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 1) & (train_data['Sex'] == 'female')

sns.distplot(train_data.loc[condition, 'Age'], label = 'Female - Survived', kde = False)

condition = (train_data['Age'].notnull()) & (train_data['Survived'] == 0) & (train_data['Sex'] == 'female')

sns.distplot(train_data.loc[condition, 'Age'], label = 'Female - Died', kde = False)

#plt.hist([train_data[(train_data['Survived'] == 1) & (train_data['Sex'] == 'female')]['Age'], train_data[(train_data['Survived'] == 0) & (train_data['Sex'] == 'female')]['Age']], stacked = True, label = ['Survived', 'Died'])

#plt.xlabel('Age')

#plt.ylabel('Number of Survived')

plt.legend();
sns.catplot(x = 'Pclass', hue = 'Embarked', col = 'Survived', data = train_data, kind = 'count')
train_data.head()
train.isna().sum()
test.isna().sum()
train.dropna(subset = ['Embarked'], axis = 0, inplace = True)
combined = train.append(test)

combined.reset_index(drop = True, inplace = True)

combined.head()
combined.isna().sum()
def getIdentity(df):

    df['Identity'] = df['Name'].map(lambda name: name.split(',')[1].split('.')[0].replace(" ",""))

    df.drop(columns = ['Name'], inplace = True)

    return df
combined = getIdentity(df = combined)
Age_dict = combined.groupby(['Sex', 'Identity']).agg('mean')['Age'].reset_index()

Age_dict['Age'] = Age_dict['Age'].map(lambda age: round(age))
Age_dict
for idx in range(len(Age_dict)):

    Sex = Age_dict.loc[idx, 'Sex']

    Identity = Age_dict.loc[idx, 'Identity']

    Age = Age_dict.loc[idx, 'Age']

    combined.loc[(combined['Age'].isna()) & (combined['Sex'] == Sex) & (combined['Identity'] == Identity), 'Age'] = Age
combined.loc[combined['Age'] <= 16, 'AgeStatus'] = 'Teenage'

combined.loc[combined['Age'] > 16, 'AgeStatus'] = 'Adult'
combined.isna().sum()
combined.groupby(['Embarked', 'Pclass']).agg('mean')['Fare']
fare_dict = combined.groupby(['Embarked', 'Pclass']).agg('mean')['Fare'].reset_index()
for idx in range(len(fare_dict)):

    embarked = fare_dict.loc[idx, 'Embarked']

    pclass = fare_dict.loc[idx, 'Pclass']

    fare = fare_dict.loc[idx, 'Fare']

    combined.loc[(combined['Fare'].isna()) & (combined['Embarked'] == embarked) & (combined['Pclass'] == pclass), 'Fare'] = fare
combined.head()
combined.isna().sum()
Identity_dict = {

    'Mr':'Local Male',

    'Mrs':'Local Female',

    'Miss':'Local Female',

    'Master':'Professional',

    'Don':'Specialist',

    'Dona':'Specialist',

    'Rev':'Professional',

    'Dr':'Professional',

    'MMe':'Foreign Female',

    'Ms':'Local Female',

    'Major':'Professional',

    'Lady':'Specialist',

    'Sir':'Specialist',

    'Mlle':'Foreign Female',

    'Col':'Specialist',

    'Capt':'Officier',

    'theCountess':'Specialist',

    'Jonkheer':'Foreign Male'

}



combined['SocialStatus'] = combined['Identity'].map(Identity_dict)
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1

combined.loc[combined['FamilySize'] == 1,'FamilyStatus'] = 'Single'

combined.loc[(combined['FamilySize'] > 1) & (combined['FamilySize'] < 4),'FamilyStatus'] = 'Normal'

combined.loc[combined['FamilySize'] > 3,'FamilyStatus'] = 'Big'
combined.loc[combined['Fare'] < 15, 'FareStatus'] = 'Ticket lower than $15'

combined.loc[combined['Fare'] >= 15, 'FareStatus'] = 'Ticket euqal/higher than $15'
combined.isna().sum()
combined.head()
combined.dropna(subset = ['SocialStatus'], axis = 0, inplace = True)
combined = combined[['PassengerId', 'Survived', 'Age', 'Sex', 'Fare', 'Embarked', 'Pclass', 'Identity', 'AgeStatus', 'SocialStatus', 'FamilySize', 'FamilyStatus', 'FareStatus']]

#combined = combined[['PassengerId', 'Survived', 'Sex', 'Embarked', 'Pclass', 'Identity', 'AgeStatus', 'SocialStatus', 'FamilySize', 'FamilyStatus', 'FareStatus']]

combined.head()
#data split

combined['Sex'] = combined['Sex'].astype('category').cat.codes

combined['Identity'] = combined['Identity'].astype('category').cat.codes

combined['Pclass'] = combined['Pclass'].astype('category').cat.codes

combined['Embarked'] = combined['Embarked'].astype('category').cat.codes

combined['AgeStatus'] = combined['AgeStatus'].astype('category').cat.codes

combined['SocialStatus'] = combined['SocialStatus'].astype('category').cat.codes

combined['FamilyStatus'] = combined['FamilyStatus'].astype('category').cat.codes

combined['FareStatus'] = combined['FareStatus'].astype('category').cat.codes

#combined['Pclass'] = combined['Pclass'].astype('category')

#combined['FamilySize'] = combined['FamilySize'].astype('category')

#combined = pd.get_dummies(data = combined)

combined['Age'] = combined['Age'].apply(lambda age: round(age))

combined['Fare'] = combined['Fare'].apply(lambda fare: round(fare, 2))



train = combined.loc[combined['Survived'].notnull()]

test = combined.loc[combined['Survived'].isna()]
train.head()
test.head()

aux = test['PassengerId']
target = train['Survived']

train.drop(columns = ['Survived'], inplace = True)

test.drop(columns = ['Survived'], inplace = True)

train_df = train.drop(columns = ['PassengerId'])

test_df = test.drop(columns = ['PassengerId'])

train_df.head()
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)



print('Parameters currently in use:\n')

print(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



print(random_grid)
rf = RandomForestClassifier()



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



rf_random.fit(train_df, target)
rf_random.best_params_
from sklearn.metrics import accuracy_score, precision_score,recall_score



def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)

    precision = precision_score(test_labels, predictions)

    recall = recall_score(test_labels, predictions)

    print('Model Performance')

    print('Accuracy = {:0.4f}%.'.format(accuracy))

    print('Precision = {:0.4f}%.'.format(precision))

    print('Recall = {:0.4f}%.'.format(recall))

    

    return accuracy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size = 0.25, random_state = 30)
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_test, y_test)
#best_random.fit(train_df, target)

output = best_random.predict(test_df).astype(int)

df_output = pd.DataFrame()

df_output['PassengerId'] = aux

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('submission.csv', index=False)