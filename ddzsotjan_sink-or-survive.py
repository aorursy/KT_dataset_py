# data handling 

import pandas as pd

from pandas import Series, DataFrame



from operator import itemgetter



# numerical manipulation and visualization

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

sns.set_style('whitegrid')

%matplotlib inline



from scipy import stats



# regular expressions

import re



# machine learning

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)
train_df.head()
train_df.info()
test_df.info()
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 3))



sns.countplot(x='Sex', data=train_df, ax=axis1)

sns.countplot(x='Survived', hue='Sex', data=train_df, ax=axis2)



fig, (axis1) = plt.subplots(1, figsize=(4, 3))

sns.barplot(x='Sex', y='Survived', data=train_df, ax=axis1)
def extract_title(name):  # function that returns a list of honorifics extracted from Names

    

    return re.findall(r',\s[\w\s]+.', name)[0][2:-1]



# Create and fill 

train_df['Title'] = train_df['Name'].map(extract_title)

test_df['Title'] = test_df['Name'].map(extract_title)



# See what kind of titles we have

print((set(train_df['Title'].values) | set(test_df['Title'].values)))
# Translating the equivalents:

train_df.loc[train_df['Title'] == 'Mlle', 'Title'] = 'Ms'

train_df.loc[train_df['Title'] == 'Miss', 'Title'] = 'Ms'

train_df.loc[train_df['Title'] == 'Mme', 'Title'] = 'Mrs'



test_df.loc[test_df['Title'] == 'Mlle', 'Title'] = 'Ms'

test_df.loc[test_df['Title'] == 'Miss', 'Title'] = 'Ms'

test_df.loc[test_df['Title'] == 'Mme', 'Title'] = 'Mrs'





# Grouping 

def assign_title(title):

    

    if title in ['Don', 'Dona', 'Lady', 'the Countess', 'Sir', 'Jonkheer']:

        return 'noble'

    elif title in ['Dr', 'Rev', 'Major', 'Col', 'Capt', 'Major']:

        return 'special'

    else:

        return title



train_df['Title'] = train_df['Title'].map(assign_title)

test_df['Title'] = train_df['Title'].map(assign_title)
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 3))



sns.barplot(x='Title', y='Survived', data=train_df, order = ['Mr', 'Master', 'Ms', 

                                                             'Mrs', 'special', 'noble'], ax=axis1)

sns.barplot(x='Title', y='Survived', data=train_df, hue='Sex', order = ['special', 'noble'], ax=axis2)





nobles = train_df[train_df['Title'] == 'special']

non_nobles = train_df[train_df['Title'] != 'special']



figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(8, 3))



sns.countplot(x='Survived', data=non_nobles, hue='Sex', ax=axis1)

sns.countplot(x='Survived', data=nobles, hue='Sex', ax=axis2)
# Filling in the missing age values with ones chosen according to a given probability distribution

def assign_age(df, age_range, probabilities):

    

    null_age_count = df['New_Age'].isnull().sum()  # number of null age values

    random_ages = np.random.choice(age_range, size=null_age_count, p=probabilities)  

                                                            # generate this number of age values

                                                            # with the given probability distribution

            

    df.loc[df['New_Age'].isnull(), 'New_Age'] = random_ages  # assign the ages to the null value entries

    

    # Check that the ages do not clash with the honorifics

    # Filters:

    is_master = df['Title'] == 'Master'

    not_master = df['Title'] != 'Master'

    not_miss = df['Title'] != 'Ms'

    is_younger = df['New_Age'] <= 16

    is_older = df['New_Age'] > 16

    

    # If there is a clash, set those age values (and only those) that clash with the honorifics to zero

    df.loc[is_master & is_older, 'New_Age'] = None

    df.loc[not_master & not_miss & is_younger, 'New_Age'] = None

    null_age_count = df['New_Age'].isnull().sum()

    

    return null_age_count  # how many New_Age entries are still null





# Estimating the kernel density for the non-null age values

train_kde = stats.gaussian_kde(train_df['Age'].dropna())

train_age_min = train_df['Age'].dropna().min().astype(int)

train_age_max = train_df['Age'].dropna().max().astype(int)



test_kde = stats.gaussian_kde(test_df['Age'].dropna())

test_age_min = test_df['Age'].dropna().min().astype(int)

test_age_max = test_df['Age'].dropna().max().astype(int)





# then build the age values to choose from and their probability distribution

train_age_points = np.array(range(train_age_min, train_age_max+1))

train_kde_points = train_kde(np.array(train_age_points))



test_age_points = np.array(range(test_age_min, test_age_max+1))

test_kde_points = test_kde(np.array(test_age_points))





# We normalise the kde to get a probability sum of 1 (discretising the continuous probability function)

train_kde_normalised = train_kde_points / (train_kde_points.sum())

test_kde_normalised = test_kde_points / (test_kde_points.sum())





null_count = train_df['Age'].isnull().sum()  # number of null age values

train_df['New_Age'] = train_df['Age']



# Now, we reassign age values according to the kde, until there are none which clash with the honorifics

while null_count:

    null_count = assign_age(train_df, train_age_points, train_kde_normalised)

    

# Doing the same for the test data

null_count = test_df['Age'].isnull().sum()  # number of null age values

test_df['New_Age'] = test_df['Age']

while null_count:

    null_count = assign_age(test_df, test_age_points, test_kde_normalised)
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 3))



fig1 = sns.distplot(train_df['Age'].dropna().astype(int), ax=axis1)

fig2 = sns.distplot(train_df['New_Age'].dropna().astype(int), color='g', ax=axis2)



fig1.set(ylim = (0, 1.3 * max(train_kde_normalised)))

fig2.set(ylim = (0, 1.3 * max(train_kde_normalised)))



# Putting the kde for both distributions in the same plot



figure, axis1 = plt.subplots(1, 1, figsize=(9, 2))



sns.kdeplot(train_df['Age'].dropna(), ax=axis1, shade=True)

sns.kdeplot(train_df['New_Age'], ax=axis1, color='g', shade=True)



# We see that the distribution is practically the same as before the imputation, and we also respected the prefixes!



# Age distribution seems fine, we give the new values to 'Age' and drop 'New_Age'

train_df['Age'] = train_df['New_Age'].astype(int)

test_df['Age'] = test_df['New_Age'].astype(int)



train_df.drop('New_Age', axis=1, inplace=True)

test_df.drop('New_Age', axis=1, inplace=True)
# Look at survival rates as a function of age

average_survival = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()



# Plotting the survival rates

sns.factorplot(x='Age', y ='Survived', kind='bar', data=average_survival, size=4, aspect=2.3)
# Binning the ages: we use the .cut() method of pandas

train_df['Age bins'] = train_df['Age']

test_df['Age bins'] = test_df['Age']



age_bins = [-0.1, 5, 16, 25, 40, 60, 100]

age_labels = ['(0-5)', '(6-16)', '(17-25)', '(26-40)', '(41-60)', '(61-)']

train_df['Age bins'] = pd.cut(train_df['Age'], age_bins, labels=age_labels)

test_df['Age bins'] = pd.cut(test_df['Age'], age_bins, labels=age_labels)



figure, axis1 = plt.subplots(1, figsize = (7, 3))

sns.barplot(x='Age bins', y='Survived', data=train_df)
train_df['FamMembers'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['FamMembers'] = test_df['SibSp'] + test_df['Parch'] + 1



sns.factorplot(x='FamMembers', data=train_df, kind='count', hue='Survived', size=4, aspect=1.7)

sns.factorplot(x='FamMembers', y='Survived', data=train_df, kind='bar', hue='Sex', size=4, aspect=1.7)
train_df.loc[train_df['FamMembers'] > 4, 'FamSize'] = 'large (>4 members)'

train_df.loc[(train_df['FamMembers'] >= 2) & (train_df['FamMembers'] <= 4), 'FamSize'] = 'small (2-4 members)'

train_df.loc[train_df['FamMembers'] == 1, 'FamSize'] = 'single'



test_df.loc[test_df['FamMembers'] > 4, 'FamSize'] = 'large (>4 members)'

test_df.loc[(test_df['FamMembers'] >= 2) & (test_df['FamMembers'] <= 4), 'FamSize'] = 'small (2-4 members)'

test_df.loc[test_df['FamMembers'] == 1, 'FamSize'] = 'single'
sns.factorplot(x='FamSize', y='Survived', data=train_df, kind='bar'

               , order = ['single', 'small (2-4 members)', 'large (>4 members)'], size = 4, aspect =1.7)
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 3))

sns.countplot(x='Pclass', data=train_df, hue='Sex', ax=axis1)

sns.barplot(x='Pclass', y='Survived', data=train_df, hue='Sex', ax=axis2)
no_fare = test_df['Fare'].isnull()

no_fare_class = test_df[no_fare]['Pclass'].values[0]



test_df.loc[no_fare, 'Fare'] = test_df[test_df['Pclass'] == no_fare_class]['Fare'].dropna().median()
train_df[train_df['Fare']<3].head()
figure, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(10,3))

sns.distplot(test_df[train_df['Pclass'] == 3]['Fare'], ax=axis1)

sns.distplot(test_df[train_df['Pclass'] == 2]['Fare'], ax=axis2)

sns.distplot(test_df[train_df['Pclass'] == 1]['Fare'], ax=axis3)



axis1.set_title('Pclass = 3', fontsize=15)

axis2.set_title('Pclass = 2', fontsize=15)

axis3.set_title('Pclass = 1', fontsize=15)
sns.factorplot(x='Pclass', y='Fare', data=train_df, hue='Survived', kind='bar', size=3, aspect=2)
train_df['Fare bins'] = train_df['Fare']

test_df['Fare bins'] = test_df['Fare']

fare_labels = ['Cheap', 'Middle', 'Expensive']



train_df['Fare bins'] = pd.qcut(train_df['Fare bins'], 3, labels=fare_labels)

test_df['Fare bins'] = pd.qcut(test_df['Fare bins'], 3, labels=fare_labels)

                                
sns.barplot(x='Fare bins', y='Survived', hue='Pclass', data=train_df)
sns.factorplot(x='FamSize', y='Fare', data=train_df, hue='Pclass', kind='bar', 

               order=['single', 'small (2-4 members)', 'large (>4 members)'], size=3, aspect=2)
sns.factorplot(x='FamMembers', y='Fare', data=train_df, hue='Pclass', kind='bar', size=3.5, aspect=2)
train_df['Fare/person'] = train_df['Fare']/train_df['FamMembers']

test_df['Fare/person'] = test_df['Fare']/test_df['FamMembers']



figure, (axis1, axis2) = plt.subplots(1, 2, figsize = (9.5,3))



fig1 = sns.kdeplot(train_df[train_df['Pclass'] == 1]['Fare/person'], 

                   shade=True, ax=axis1, label= 'Pclass = 1')

fig2 = sns.kdeplot(train_df[train_df['Pclass'] == 2]['Fare/person'], 

                   shade=True, ax=axis1, label= 'Pclass = 2')

fig3 = sns.kdeplot(train_df[train_df['Pclass'] == 3]['Fare/person'], 

                   shade=True, ax=axis1, label= 'Pclass = 3')



fig1.set(xlim = (0, 100))

fig1.set_xlabel('Fare/person')

fig2.set(xlim = (0, 100))

fig3.set(xlim = (0, 100))



sns.barplot(x='Pclass', y='Fare/person', data=train_df, hue='Survived', ax=axis2)
train_df['Fare/person bins'] = train_df['Fare/person']

test_df['Fare/person bins'] = test_df['Fare/person']



fpp_bins = [-0.1, 20, 40, 60, 80, 100, 150, 600]

fpp_labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '101-150', '151-']

train_df['Fare/person bins'] = pd.cut(train_df['Fare/person'], fpp_bins, labels=fpp_labels)

test_df['Fare/person bins'] = pd.cut(test_df['Fare/person'], fpp_bins, labels=fpp_labels)



sns.factorplot(x='Fare/person bins', y='Survived', hue='Pclass', data=train_df, kind='bar', 

               size=3, aspect=2.5)
train_df.info()
train_df[train_df['Embarked'].isnull()]
median_S = train_df[(train_df['Embarked'] == 'S') &(train_df['Pclass'] == 1)]['Fare'].median()

median_C = train_df[(train_df['Embarked'] == 'C') &(train_df['Pclass'] == 1)]['Fare'].median()

median_Q = train_df[(train_df['Embarked'] == 'Q') &(train_df['Pclass'] == 1)]['Fare'].median()



figure, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(11, 3))

sns.distplot(train_df[(train_df['Embarked'] == 'S') & (train_df['Pclass'] == 1)]['Fare'], ax=axis1)

sns.distplot(train_df[(train_df['Embarked'] == 'C') & (train_df['Pclass'] == 1)]['Fare'], ax=axis2)

fig=sns.distplot(train_df[(train_df['Embarked'] == 'Q') & (train_df['Pclass'] == 1)]['Fare'], ax=axis3)

fig.set(xlim=(80, 100))



axis1.set_title('Southampton; median: ' + str(median_S), fontsize = 11)

axis2.set_title('Cherbourg; median: ' + str(median_C), fontsize = 11)

axis3.set_title('Queenstown; median: ' + str(median_Q), fontsize = 11)
train_df.loc[train_df['Embarked'].isnull(), 'Embarked'] = 'C'
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 3))



sns.countplot(x='Embarked', hue='Survived', data=train_df, ax=axis1)

sns.barplot(x='Embarked', y='Survived', data=train_df, ax=axis2)
train_df['Age*Pclass'] = train_df['Age'] * train_df['Pclass']

test_df['Age*Pclass'] = test_df['Age'] * test_df['Pclass']



train_df['Age*Pclass bins'] = pd.qcut(train_df['Age*Pclass'], 6)







test_df['Age*Pclass bins'] = pd.qcut(test_df['Age*Pclass'], 6)



sns.factorplot(x='Age*Pclass bins', y='Survived', data = train_df, kind='bar', size=3, aspect=2.3)
train_df.info()
# Initialising the training and test data

X_train = pd.DataFrame({'Sex': train_df['Sex']})



Y_train = train_df['Survived']



X_test = pd.DataFrame({'Sex': test_df['Sex']})  
# We numerise the string values for the analysis - not all possible variables are used, feel free to play around with them!



# Sex:

X_train['Sex'] = train_df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)

X_test['Sex'] = test_df['Sex'].map({'female' : 0, 'male' : 1}).astype(int)



# Embarkation:

X_train['Embarked'] = train_df['Embarked'].map({'S' : 0, 'Q' : 1, 'C' : 2}).astype(int)

X_test['Embarked'] = test_df['Embarked'].map({'S' : 0, 'Q' : 1, 'C' : 2}).astype(int)



# Title

X_train['Title'] = train_df['Title'].map({'Mr':0, 'Master':1, 'Ms':2, 'Mrs':3, 'special':4, 'noble':5}).astype(int)

X_test['Title'] = test_df['Title'].map({'Mr':0, 'Master':1, 'Ms':2, 'Mrs':3, 'special':4, 'noble':5}).astype(int)



# Age:

X_train['Age'] = train_df['Age']

X_test['Age'] = test_df['Age']



# new_labels = [label for label in range(len(age_labels))]

# label_dict = dict(zip(age_labels, new_labels))



# X_train['Age bins'] = train_df['Age bins'].map(label_dict).astype(int)

# X_test['Age bins'] = test_df['Age bins'].map(label_dict).astype(int)





# Family size:

famsize_labels = list(train_df['FamSize'].unique())

new_labels = [label for label in range(len(famsize_labels))]

label_dict = dict(zip(famsize_labels, new_labels))



X_train[ 'FamSize'] = train_df['FamSize'].map(label_dict).astype(int)

X_test['FamSize'] = test_df['FamSize'].map(label_dict).astype(int)





# Fare:

X_train['Fare'] = train_df['Fare']

X_test['Fare'] = test_df['Fare']



# new_labels = [label for label in range(len(fare_labels))]

# label_dict = dict(zip(fare_labels, new_labels))



# X_train['Fare bins'] = train_df['Fare bins'].map(label_dict).astype(int)

# X_test['Fare bins'] = test_df['Fare bins'].map(label_dict).astype(int)





# Parch:

X_train['Parch'] = train_df['Parch']

X_test['Parch'] = test_df['Parch']





# SibSp:

X_train['SibSp'] = train_df['SibSp']

X_test['SibSp'] = test_df['SibSp']





# Fare/person:

X_train['Fare/person'] = train_df['Fare/person']

X_test['Fare/person'] = test_df['Fare/person']



# new_labels = [label for label in range(len(fpp_labels))]

# label_dict = dict(zip(fpp_labels, new_labels))



# X_train['Fare/person bins'] = train_df['Fare/person bins'].map(label_dict).astype(int)

# X_test['Fare/person bins'] = test_df['Fare/person bins'].map(label_dict).astype(int)





# Age*Pclass:

X_train['Age*Pclass'] = train_df['Age*Pclass']

X_test['Age*Pclass'] = test_df['Age*Pclass']



# apc_labels = list(train_df['Age*Pclass bins'].unique())

# new_labels = [label for label in range(len(apc_labels))]

# label_dict = dict(zip(apc_labels, new_labels))

# X_train['Age*Pclass bins'] = train_df['Age*Pclass bins'].map(label_dict).astype(int)



# apc_labels = list(test_df['Age*Pclass bins'].unique())

# new_labels = [label for label in range(len(apc_labels))]

# label_dict = dict(zip(apc_labels, new_labels))

# X_test['Age*Pclass bins'] = test_df['Age*Pclass bins'].map(label_dict).astype(int)
X_train.head(2)
X_train_np = X_train.values

Y_train_np = Y_train.values



X_test_np = X_test.values
random_forests = RandomForestClassifier(n_estimators=50)

random_forests.fit(X_train_np, Y_train_np)
Y_pred_np = random_forests.predict(X_test_np)

random_forests.score(X_train_np, Y_train_np)
predictors = list(X_train.columns.values)

relevance = list(random_forests.feature_importances_)



# in order to sort them in descending relevance we combine, sort, then split it again into two lists

combined_list = list(zip(predictors, relevance))

sorted_list = sorted(combined_list, key=itemgetter(1), reverse=True)

unzipped = list(zip(*sorted_list))



predictors = list(unzipped[0])

relevance = list(unzipped[1])



# Plotting the predictor relevance

predictor_relevance = pd.DataFrame({

        'Predictors' : predictors,

        'Relevance' : relevance

    })





# sns.set(font_scale=1.3)

sns.set_style('whitegrid')

sns.factorplot(x='Relevance', y='Predictors', data=predictor_relevance, 

               palette='GnBu_r', kind='bar', size=4, aspect=1.5)
# Creating the submission csv

submission = pd.DataFrame({

        'PassengerId' : test_df['PassengerId'],

        'Survived' : Y_pred_np

    })



submission.to_csv('sinkorsurvive.csv', index=False)