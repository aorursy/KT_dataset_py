# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Imports

import matplotlib.pyplot as plt

import seaborn as sns



import string

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



import warnings

warnings.filterwarnings('ignore')



SEED = 2019
# helper Functions



def list_helpers():

    """ Prints all helper functions available in current notebook """

    print(' missing_percentage(data)')



def missing_percentage(data):

    """

    Prints the count of missing values and overall percentage missing for each feature

    """

    rows, cols = data.shape

    num_missing = data.isnull().sum()

    missing_percent = (((data.isnull().sum())/data.shape[0]) * 100)

    print(pd.concat([num_missing, missing_percent], axis=1).rename(columns={0:'Num_Missing',1:'Missing_Percent'}).sort_values(by='Missing_Percent', ascending=False))

    

def feature_correlations(corr, feature):

    """

    Return a series containing all correlations with a given feature 

    """

    return corr[corr['Feature 1'] == feature]

    



def concat_df(train, test):

    return pd.concat([train, test], sort=True).reset_index(drop=True)



def divide_df(all_data, split):

    # Returns divided dfs of training and test set

    return all_data.loc[:split-1], all_data.loc[split:].drop(['Survived'], axis=1)
# Read Data

train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col=0)

test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col=0)

full = concat_df(train, test)

dfs = [train, test]



split = train.shape[0]



# Constructs a two index dataframe containing the correlation between two features

corr = full.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)



print('Number of Training Examples = {}'.format(train.shape[0]))

print('Number of Test Examples = {}\n'.format(test.shape[0]))

print('Training X Shape = {}'.format(train.shape))

print('Training y Shape = {}\n'.format(train['Survived'].shape[0]))

print('Test X Shape = {}'.format(test.shape))

print('Test y Shape = {}\n'.format(test.shape[0]))

print(train.columns)

print(test.columns)
print(train.info())
train.sample(3)
print(test.info())
test.sample(3)
print(missing_percentage(full))
print("{} \n\n {}".format(feature_correlations(corr, 'Age'), full.groupby(['Sex','Pclass']).median()['Age']))
# Filling the missing values in Age with the medians of Sex and Pclass groups

full['Age'] = np.where((full.Sex == 'female') & (full.Pclass == 1) & (full.Age.isnull()), 36,

                       np.where((full.Sex == 'female') & (full.Pclass == 2) & (full.Age.isnull()), 28,

                           np.where((full.Sex == 'female') & (full.Pclass == 3) & (full.Age.isnull()), 22,

                               np.where((full.Sex == 'male') & (full.Pclass == 1) & (full.Age.isnull()), 42,

                                   np.where((full.Sex == 'male') & (full.Pclass == 2) & (full.Age.isnull()), 29.5,

                                       np.where((full.Sex == 'male') & (full.Pclass == 3) & (full.Age.isnull()), 25, full.Age))))))
full[full['Embarked'].isnull()]
full['Embarked'].fillna('S', inplace=True)
full['Fare'].fillna(full.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0], inplace=True)
# Replace all numerical Cabins with 'M'

full['Deck'] = full['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
# The distribution of Decks and count of each Pclass per deck; it's a two level dataframe with Deck and Pclass level 0 and level 1 respectively

full_decks = full.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()



# Dictionary to track the number of passengers belongin to each group

deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}



# full_decks is a two index df, we select the first level with .levels[0] which returns 'Deck'

decks = full_decks.columns.levels[0]



full_decks
# For each deck, we update deck counts with the count of passengers per Pclass

for deck in decks:

    for pclass in range(1, 4):

        try:

            count = full_decks[deck][pclass][0]

            deck_counts[deck][pclass] = count 

        except KeyError:

            deck_counts[deck][pclass] = 0



deck_dist = pd.DataFrame(deck_counts)    

deck_dist
# The distribution percetnage of Pclass 1, 2, and 3

deck_percentages = {}

for col in deck_dist.columns:

        deck_percentages[col] = [(count / deck_dist[col].sum()) * 100 for count in deck_dist[col]]



deck_percentages
Pclass1, Pclass2, Pclass3 = [],[],[]

for deck in deck_percentages:

    Pclass1.append(deck_percentages[deck][0])

    

for deck in deck_percentages:

    Pclass2.append(deck_percentages[deck][0] + deck_percentages[deck][1])

    

for deck in deck_percentages:

    Pclass3.append(deck_percentages[deck][0] + deck_percentages[deck][1] + deck_percentages[deck][2])

    

plt.figure(figsize=(16,6))

sns.barplot(x=deck_dist.columns, y=Pclass3, label="Pclass3", color="#2B95DB")

sns.barplot(x=deck_dist.columns, y=Pclass2, label="Pclass2", color="#5EC4FF")

sns.barplot(x=deck_dist.columns, y=Pclass1, label="Pclass1", color="#9BD8FF")





plt.title("sdf")

plt.xlabel("Decks")

plt.legend()
full[full['Deck'] == 'T']
full.Deck = np.where((full.Deck == 'T'),'A',full.Deck)
full.groupby(['Deck','Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'Ticket', 'Pclass']).rename(columns={'Name': 'Count'}).transpose()        

deck_surv = full.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'Ticket']).rename(columns={'Name':'Count'}).transpose()

deck_surv
df_decks_percent = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}}



for deck in deck_surv.columns.levels[0]:

    df_decks_percent[deck] = deck_surv[deck][0].sum() / (deck_surv[deck][0].sum() + deck_surv[deck][1].sum())



survived,not_survived = [], []



for key in df_decks_percent:

    survived.append(df_decks_percent[key])

    not_survived.append(1)



plt.figure(figsize=(16,6))

sns.barplot(x=deck_surv.columns.levels[0], y=not_survived, label="not survived", color="#9BD8FF")

sns.barplot(x=deck_surv.columns.levels[0], y=survived, label="survived", color="#2B95DB")

plt.legend()
print(deck_dist)

full['Deck'].replace(['A', 'B', 'C'], 'ABC', inplace=True)

full['Deck'].replace(['D', 'E'], 'DE', inplace=True)

full['Deck'].replace(['F', 'G'], 'FG', inplace=True)

full['Deck'].value_counts()
full.drop(['Cabin'], axis = 1, inplace=True)
missing_percentage(full)
train, test = divide_df(full, split)
print(train['Survived'].value_counts())

survived = train['Survived'].value_counts()[1]

not_survived = train['Survived'].value_counts()[0]
sns.barplot(x=['Survived','Not Survived'], y=[survived, not_survived])
train_corr = train.corr().abs().unstack().sort_values(ascending=False).reset_index()

train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

train_corr.drop(train_corr[train_corr['Correlation Coefficient'] == 1.0].index, inplace=True)

train_corr[train_corr['Correlation Coefficient'] > .1]
plt.figure(figsize=(16,6))

sns.heatmap(train.corr(), annot=True, square=True, cmap='coolwarm', annot_kws={'size': 12})
cont_features = ['Age', 'Fare']

surv = train['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(cont_features):    

    # Distribution of survival in feature

    sns.distplot(train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])

    sns.distplot(train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    

    # Distribution of feature in dataset

    sns.distplot(train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

    

    for j in range(2):        

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

    

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

        

plt.show()

# Cut the Fare into 13 bins (we normally would normally not want a bin count > 15)

full['Fare'] = pd.qcut(full['Fare'], 13)



plt.figure(figsize=(22,9))



# We plot the Fare with respect to those who survived and did not survive

sns.countplot(x='Fare', hue='Survived', data=full)



# labels

plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)
# There are 99 unique values, being close to 100 I cut Age into 10 bins

full['Age'] = pd.qcut(full['Age'], 10)



plt.figure(figsize=(22,9))



# We plot the Age with respect to those who survived and did not survive

sns.countplot(x='Age', hue='Survived', data=full)



plt.ylim=(0,)
full.Age.value_counts()
# Look at the survival and non-survival rate distributions with respect to each feature 

cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
full = concat_df(train, test)
full['Fare'] = pd.qcut(full['Fare'], 13)



fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Fare', hue='Survived', data=full)



plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)



plt.show()
full['Age'] = pd.qcut(full['Age'], 10)



fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Age', hue='Survived', data=full)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)



plt.show()
full['Family_Size'] = full['SibSp'] + full['Parch'] + 1



fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=full['Family_Size'].value_counts().index, y=full['Family_Size'].value_counts().values, ax=axs[0][0])

sns.countplot(x='Family_Size', hue='Survived', data=full, ax=axs[0][1])



axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)



family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

full['Family_Size_Grouped'] = full['Family_Size'].map(family_map)



sns.barplot(x=full['Family_Size_Grouped'].value_counts().index, y=full['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])

sns.countplot(x='Family_Size_Grouped', hue='Survived', data=full, ax=axs[1][1])



axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)

axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)



for i in range(2):

    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        axs[i][j].set_xlabel('')

        axs[i][j].set_ylabel('')



plt.show()
full['Ticket_Frequency'] = full.groupby('Ticket')['Ticket'].transform('count')



fig, axs = plt.subplots(figsize=(12, 9))

sns.countplot(x='Ticket_Frequency', hue='Survived', data=full)



plt.xlabel('Ticket Frequency', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)



plt.show()
full['Title'] = full['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



fig, axs = plt.subplots(nrows=2, figsize=(20, 12))





sns.barplot(x=full['Title'].value_counts().index, y=full['Title'].value_counts().values, ax=axs[0])



axs[0].tick_params(axis='x', labelsize=10)

axs[1].tick_params(axis='x', labelsize=15)



for i in range(2):    

    axs[i].tick_params(axis='y', labelsize=15)



axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)



# Reduce cardinality of Title

full['Title'] = full['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

full['Title'] = full['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



sns.barplot(x=full['Title'].value_counts().index, y=full['Title'].value_counts().values, ax=axs[1])

axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)



plt.show()
full['Is_Married'] = 0

full['Is_Married'].loc[full['Title'] == 'Mrs'] = 1
def extract_surname(data):    

    

    families = []

    

    for i in range(len(data)):        

        name = data.iloc[i]

        

        if '(' in name:

            name_no_bracket = name.split('(')[0] 

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        for c in string.punctuation:

            family = family.replace(c, '').strip()

            

        families.append(family)

            

    return families



full['Family'] = extract_surname(full['Name'])

train = full.loc[:890]

test = full.loc[891:]

dfs = [train, test]
# Creating a list of families and tickets that are occuring in both training and test set

non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]

non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]





df_family_survival_rate = train.groupby('Family')['Survived', 'Family','Family_Size'].median()

df_ticket_survival_rate = train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()



family_rates = {}

ticket_rates = {}



# Checking a family exists in both training and test set, and has members more than 1

# If family is in train and test and more than 1 individual

for i in range(len(df_family_survival_rate)):

    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:

        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

        

# Checking a ticket exists in both training and test set, and has members more than 1

# If ticket is in train and test and more than 1 occurence

for i in range(len(df_ticket_survival_rate)):    

    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:

        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
# Overall mean survival rate of the training set

mean_survival_rate = np.mean(train['Survived'])



train_family_survival_rate = []

train_family_survival_rate_NA = []

test_family_survival_rate = []

test_family_survival_rate_NA = []



# For each family in train , if it's in family rates, append the family size to family survival rate. Also, append 1 if present

# If not present, append mean survival rate and 0

for i in range(len(train)):

    if train['Family'][i] in family_rates:

        train_family_survival_rate.append(family_rates[train['Family'][i]])

        train_family_survival_rate_NA.append(1)

    else:

        train_family_survival_rate.append(mean_survival_rate)

        train_family_survival_rate_NA.append(0)

    

# For each family in test , if it's in family rates, append the family size to family survival rate. Also, append 1 if present

# If not present, append mean survival rate and 0

for i in range(len(test)):

    if test['Family'].iloc[i] in family_rates:

        test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])

        test_family_survival_rate_NA.append(1)

    else:

        test_family_survival_rate.append(mean_survival_rate)

        test_family_survival_rate_NA.append(0)

        

# Add these survival rates to both train and test data

train['Family_Survival_Rate'] = train_family_survival_rate

train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA

test['Family_Survival_Rate'] = test_family_survival_rate

test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA



train_ticket_survival_rate = []

train_ticket_survival_rate_NA = []

test_ticket_survival_rate = []

test_ticket_survival_rate_NA = []



# For each ticket in train , if it's in ticket rates, append the ticket frequency size to ticket survival rate. Also, append 1 if present

# If not present, append mean survival rate and 0

for i in range(len(train)):

    if train['Ticket'][i] in ticket_rates:

        train_ticket_survival_rate.append(ticket_rates[train['Ticket'][i]])

        train_ticket_survival_rate_NA.append(1)

    else:

        train_ticket_survival_rate.append(mean_survival_rate)

        train_ticket_survival_rate_NA.append(0)



# For each ticket in test , if it's in ticket rates, append the ticket frequency size to ticket survival rate. Also, append 1 if present

# If not present, append mean survival rate and 0        

for i in range(len(test)):

    if test['Ticket'].iloc[i] in ticket_rates:

        test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])

        test_ticket_survival_rate_NA.append(1)

    else:

        test_ticket_survival_rate.append(mean_survival_rate)

        test_ticket_survival_rate_NA.append(0)



# Add these survival rates to both train and test data

train['Ticket_Survival_Rate'] = train_ticket_survival_rate

train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA

test['Ticket_Survival_Rate'] = test_ticket_survival_rate

test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
dfs = [train, test]
# Let survival rates for train and test sets be the average of both ticket and family survival rates

for df in dfs:

    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2

    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2    
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:

    for feature in non_numeric_features:     

        df[feature] = LabelEncoder().fit_transform(df[feature])
cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']

encoded_features = []



for df in dfs:

    for feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)



train = pd.concat([train, *encoded_features[:6]], axis=1)

test = pd.concat([test, *encoded_features[6:]], axis=1)
full = concat_df(train, test)

drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',

             'Name', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',

            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']



full.drop(columns=drop_cols, inplace=True)
X_train = StandardScaler().fit_transform(train.drop(columns=drop_cols))

y_train = train['Survived'].values

X_test = StandardScaler().fit_transform(test.drop(columns=drop_cols))

y_test = test['Survived'].values



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))

print('y_test shape: {}'.format(y_test.shape))
"""

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



pipe = Pipeline(steps=[('classifier', RandomForestClassifier(random_state = 2019))])



param_grid = { 

    'classifier__criterion': ['gini','entropy'],

    'classifier__n_estimators': list(range(1,10,1)) + list(range(10,50,10)) + list(range(50,2001,50)),

    'classifier__max_depth': range(1,11,1),

    'classifier__min_samples_split': range(2,5,1),

    'classifier__min_samples_leaf': range(2,11,1),

    'classifier__max_features': ['auto', 'sqrt', 'log2'],

    'classifier__oob_score': [True],

    'classifier__verbose': range(0,5,1)}





CV = GridSearchCV(pipe, param_grid, n_jobs= 1)

CV.fit(X_train, y_train)  

print(CV.best_params_)    

print(CV.best_score_)

"""
single_best_model = RandomForestClassifier(criterion='gini', 

                                           n_estimators=1100, # The number of trees in the forest.

                                           max_depth=5, #The maximum depth of the tree 

                                           min_samples_split=4, # The minimum number of samples required to split an internal node:

                                           min_samples_leaf=5, # The minimum number of samples required to be at a leaf node. A split point at any depth will 

                                                               # only be considered if it leaves at least min_samples_leaf training samples in each of the 

                                                               # left and right branches. This may have the effect of smoothing the model, especially in

                                                               # regression.

                                           max_features='auto', # The number of features to consider when looking for the best split:

                                           oob_score=True, # Whether to use out-of-bag samples to estimate the generalization accuracy.

                                           random_state=SEED,

                                           n_jobs=-1, # -1 means use all processors

                                           verbose=1)



leaderboard_model = RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=SEED,

                                           n_jobs=-1,

                                           verbose=1) 
N = 5 # Number of golds

oob = 0 # out of bag score



# Creates a table to contain each fold probability

probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])

importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=full.columns)

fprs, tprs, scores = [], [], []



# Provides train/test indices to split data in train/test sets.

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)



# For enumerate, the second parameter returns the fold number

# skf -> provides train/test indices to split data in train/test sets.

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    print('Fold {}\n'.format(fold))

    

    # Fitting the model

    leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])

    

    # Computing Train AUC score

    # roc_curve takes y_true binary labels, y_scores and returns false positive rate, true positive rate, and decreasing thresholds

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], leaderboard_model.predict_proba(X_train[trn_idx])[:, 1])

    trn_auc_score = auc(trn_fpr, trn_tpr)

    

    # Computing Validation AUC score

    val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], leaderboard_model.predict_proba(X_train[val_idx])[:, 1])

    val_auc_score = auc(val_fpr, val_tpr)  

      

    # Append scores

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)

    

    # X_test probabilities

    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]

    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]

    

    # Sets all rows in fold-1 column

    importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_

        

    oob += leaderboard_model.oob_score_ / N

    print('Fold {} OOB Score: {}\n'.format(fold, leaderboard_model.oob_score_))   

    

print('Average OOB Score: {}'.format(oob))
importances['Mean_Importance'] = importances.mean(axis=1)

importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)



plt.figure(figsize=(15, 20))

sns.barplot(x='Mean_Importance', y=importances.index, data=importances)



plt.xlabel('')

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)

plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)



plt.show()
def plot_roc_curve(fprs, tprs):

    

    tprs_interp = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    f, ax = plt.subplots(figsize=(15, 15))

    

    # Plotting ROC for each fold and computing AUC scores

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):

        # mean_fpr -> np.linspace -> The x-coordinates at which to evaluate the interpolated values.

        # fpr -> x -> The x-coordinates of the data points, must be increasing if argument period is not specified

        # tpr -> y -> The y-coordinates of the data points, same length as xp.

        # interp -> returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

        tprs_interp.append(np.interp(mean_fpr, fpr, tpr)) # line for each fold

        tprs_interp[-1][0] = 0.0

        roc_auc = auc(fpr, tpr) # auc score for fold i

        aucs.append(roc_auc) # append to list of scores

        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc)) # plot each fold

        

    # Plotting ROC for random guessing

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    

    mean_tpr = np.mean(tprs_interp, axis=0) # mean of all lines

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr) # mean scores

    std_auc = np.std(aucs) # std of scores

    

    # Plotting the mean ROC

    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)

    

    # Plotting the standard deviation around the mean ROC Curve

    std_tpr = np.std(tprs_interp, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

    

    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)

    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)

    ax.tick_params(axis='x', labelsize=15)

    ax.tick_params(axis='y', labelsize=15)

    ax.set_xlim([-0.05, 1.05])

    ax.set_ylim([-0.05, 1.05])



    ax.set_title('ROC Curves of Folds', size=20, y=1.02)

    ax.legend(loc='lower right', prop={'size': 13})

    

    plt.show()



plot_roc_curve(fprs, tprs)
# Gets all survived columns

class_survived = [col for col in probs.columns if col.endswith('Prob_1')]



# Sums all columns for fold 1, and divide by N to get average

probs['1'] = probs[class_survived].sum(axis=1) / N

probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / N



# Add new column called pred

probs['pred'] = 0



# Get index of all probabilities >= .5

pos = probs[probs['1'] >= 0.5].index



# Locate all indexes that match pos and set 'pred' = 1

probs.loc[pos, 'pred'] = 1



# Use probs['pred'], convert to int, and use them to predict

y_pred = probs['pred'].astype(int)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])



# Set PassengerId to text.index + 1

submission_df['PassengerId'] = test.index + 1



# Set survived to y_pred values

submission_df['Survived'] = y_pred.values

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)