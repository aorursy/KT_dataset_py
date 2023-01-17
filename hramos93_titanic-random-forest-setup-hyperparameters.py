import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from numpy import mean

from numpy import std

import string

import warnings
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
dfTrain = pd.read_csv(dirname+"/train.csv")

dfTest = pd.read_csv(dirname+"/test.csv")

dfGenderSubmission = pd.read_csv(dirname+"/gender_submission.csv")
dfTrain.head(5)
dfTest.head(5)
dfGenderSubmission.head(5)
print('The columns of Train are:\n ', dfTrain.columns, 'and are', len(dfTrain.columns),'columns \n', '='*90,'\n', 

            '\n','The columns of Test are: \n', dfTest.columns,'and are',len(dfTest.columns),'columns'  )


def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

dfMrg = concat_df(dfTrain,dfTest)

dfMrg.isna().sum()
dfMrg.describe()
dfMrg['Title'] = dfMrg['Name'].str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(dfMrg['Title'],dfMrg['Sex'])
dfMrg['Title'] = dfMrg['Title'].replace(['Mlle','Lady','Dona','Ms','Countess'],'Miss')

dfMrg['Title'] = dfMrg['Title'].replace(['Dona','Mme'],'Mrs')

dfMrg['Title'] = dfMrg['Title'].replace(['Don','Jonkheer','Sir','Master'],'Mr')

dfMrg['Title'] = dfMrg['Title'].replace(['Capt','Col','Major','Rev'],'Crew')

pd.crosstab(dfMrg['Title'],dfMrg['Sex'])
df_all_corr = dfMrg.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'Age']
#Replace title Mr and Miss to Youth for people less than 15 years old

titleyouth = dfMrg[(dfMrg['Title'] == 'Miss') | (dfMrg['Title'] == 'Mr')]

dfMrg.loc[titleyouth[(titleyouth['Age'] < 21)].index.tolist(),'Title'] = 'Youth'
#Replace title Dr to Mr or Mrs. 

titleDr = dfMrg[(dfMrg['Title'] == 'Dr')].dropna()

dfMrg.loc[titleDr[(titleDr['Sex'] == 'female')].index.tolist(),'Title'] = 'Mrs'

dfMrg['Title'] = dfMrg['Title'].replace(['Dr'],'Mr')



pd.crosstab(dfMrg['Title'],dfMrg['Sex'])


dfMrg.set_index('Title').isna().sum(level=0)['Age']
age_by_pclass_sex = dfMrg.groupby(['Sex', 'Pclass']).median()['Age']
for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(dfMrg['Age'].median()))
dfMrg['Age'] = dfMrg.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
sns.distplot(dfMrg['Age'].dropna(), hist=True, kde=True, 

              color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

plt.title("Age Distribution")
dfMrg[dfMrg['Embarked'].isnull()]

dfMrg['Embarked'] = dfMrg['Embarked'].fillna('S')
dfMrg[dfMrg['Fare'].isnull()]

med_fare = dfMrg.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]



dfMrg['Fare'] = dfMrg['Fare'].fillna(med_fare)
dfMrg['Deck'] = dfMrg['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df_all_decks = dfMrg.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 

                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()
def get_pclass_dist(df):

    

    # Creating a dictionary for every passenger class count in every deck

    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}

    decks = df.columns.levels[0]    

    

    for deck in decks:

        for pclass in range(1, 4):

            try:

                count = df[deck][pclass][0]

                deck_counts[deck][pclass] = count 

            except KeyError:

                deck_counts[deck][pclass] = 0

                

    df_decks = pd.DataFrame(deck_counts)    

    deck_percentages = {}



    # Creating a dictionary for every passenger class percentage in every deck

    for col in df_decks.columns:

        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

        

    return deck_counts, deck_percentages







def display_pclass_dist(percentages):

    

    df_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85

    

    pclass1 = df_percentages[0]

    pclass2 = df_percentages[1]

    pclass3 = df_percentages[2]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')

    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')

    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')



    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)   

    

    plt.show()  
all_deck_count, all_deck_per = get_pclass_dist(df_all_decks)

display_pclass_dist(all_deck_per)
# Passenger in the T deck is changed to A

idx = dfMrg[dfMrg['Deck'] == 'T'].index

dfMrg.loc[idx, 'Deck'] = 'A'
df_all_decks_survived = dfMrg.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 

                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()
def get_survived_dist(df):

    

    # Creating a dictionary for every survival count in every deck

    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}

    decks = df.columns.levels[0]    



    for deck in decks:

        for survive in range(0, 2):

            surv_counts[deck][survive] = df[deck][survive][0]

            

    df_surv = pd.DataFrame(surv_counts)

    surv_percentages = {}



    for col in df_surv.columns:

        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

        

    return surv_counts, surv_percentages



def display_surv_dist(percentages):

    

    df_survived_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85    



    not_survived = df_survived_percentages[0]

    survived = df_survived_percentages[1]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")

    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")

 

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    

    plt.show()



all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)

display_surv_dist(all_surv_per)
dfMrg['Deck'] = dfMrg['Deck'].replace(['A', 'B', 'C'], 'ABC')

dfMrg['Deck'] = dfMrg['Deck'].replace(['D', 'E'], 'DE')

dfMrg['Deck'] = dfMrg['Deck'].replace(['F', 'G'], 'FG')
dfMrg['Deck'].value_counts()
dfMrg.drop(['Cabin'], inplace=True, axis=1)
df_train, df_test = divide_df(dfMrg)

dfs = [df_train, df_test]
def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in dfs:

    display_missing(df)
cont_features = ['Age', 'Fare']

surv = df_train['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(cont_features):    

    # Distribution of survival in feature

    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])

    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    

    # Distribution of feature in dataset

    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    

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
dfMrg = concat_df(df_train, df_test)

dfMrg.head()
dfMrg['Fare'] = pd.qcut(dfMrg['Fare'], 13)
fig, asx = plt.subplots(figsize=(22, 9))

sns.countplot(x= 'Fare', hue='Survived', data=dfMrg)

plt.xlabel('Fare', size = 15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)



plt.show()
dfMrg['Age'] = pd.qcut(dfMrg['Age'],10)
fig, axs = plt.subplots(figsize=(22,9))

sns.countplot(x='Age', hue='Survived', data=dfMrg)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passeger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)



plt.show()
dfMrg['Family_Size'] = dfMrg['SibSp'] + dfMrg['Parch'] + 1



fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=dfMrg['Family_Size'].value_counts().index, y=dfMrg['Family_Size'].value_counts().values, ax=axs[0][0])

sns.countplot(x='Family_Size', hue='Survived', data=dfMrg, ax=axs[0][1])



axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)



family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

dfMrg['Family_Size_Grouped'] = dfMrg['Family_Size'].map(family_map)



sns.barplot(x=dfMrg['Family_Size_Grouped'].value_counts().index, y=dfMrg['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])

sns.countplot(x='Family_Size_Grouped', hue='Survived', data=dfMrg, ax=axs[1][1])



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
dfMrg['Ticket_Frequency'] = dfMrg.groupby('Ticket')['Ticket'].transform('count')
fig, axs = plt.subplots(figsize=(12, 9))

sns.countplot(x='Ticket_Frequency', hue='Survived', data=dfMrg)



plt.xlabel('Ticket Frequency', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size':15})

plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y = 1.05)



plt.show()
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



dfMrg['Family'] = extract_surname(dfMrg['Name'])

df_train = dfMrg.loc[:890]

df_test = dfMrg.loc[891:]

dfs = [df_train, df_test]


non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]

non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]



df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()

df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()



family_rates = {}

ticket_rates = {}



for i in range(len(df_family_survival_rate)):

    # Checking a family exists in both training and test set, and has members more than 1

    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:

        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]



for i in range(len(df_ticket_survival_rate)):

    # Checking a ticket exists in both training and test set, and has members more than 1

    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:

        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
mean_survival_rate = np.mean(df_train['Survived'])



train_family_survival_rate = []

train_family_survival_rate_NA = []

test_family_survival_rate = []

test_family_survival_rate_NA = []



for i in range(len(df_train)):

    if df_train['Family'][i] in family_rates:

        train_family_survival_rate.append(family_rates[df_train['Family'][i]])

        train_family_survival_rate_NA.append(1)

    else:

        train_family_survival_rate.append(mean_survival_rate)

        train_family_survival_rate_NA.append(0)

        

for i in range(len(df_test)):

    if df_test['Family'].iloc[i] in family_rates:

        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])

        test_family_survival_rate_NA.append(1)

    else:

        test_family_survival_rate.append(mean_survival_rate)

        test_family_survival_rate_NA.append(0)

        

df_train['Family_Survival_Rate'] = train_family_survival_rate

df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA

df_test['Family_Survival_Rate'] = test_family_survival_rate

df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA



train_ticket_survival_rate = []

train_ticket_survival_rate_NA = []

test_ticket_survival_rate = []

test_ticket_survival_rate_NA = []



for i in range(len(df_train)):

    if df_train['Ticket'][i] in ticket_rates:

        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])

        train_ticket_survival_rate_NA.append(1)

    else:

        train_ticket_survival_rate.append(mean_survival_rate)

        train_ticket_survival_rate_NA.append(0)

        

for i in range(len(df_test)):

    if df_test['Ticket'].iloc[i] in ticket_rates:

        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])

        test_ticket_survival_rate_NA.append(1)

    else:

        test_ticket_survival_rate.append(mean_survival_rate)

        test_ticket_survival_rate_NA.append(0)

        

df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate

df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA

df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate

df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
for df in [df_train, df_test]:

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



df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)

df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)
dfMrg = concat_df(df_train, df_test)

drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',

             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',

            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']



dfMrg.drop(columns=drop_cols, inplace=True)



dfMrg.head()
X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))

y_train = df_train['Survived'].values

X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))

y_test = dfGenderSubmission.drop("PassengerId", axis=1).copy()
SEED = 42

single_best_model = RandomForestClassifier(criterion='gini', 

                                           n_estimators=1100,

                                           max_depth=5,

                                           min_samples_split=4,

                                           min_samples_leaf=5,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=SEED,

                                           n_jobs=-1,

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
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold

N = 5

oob = 0

probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])

importances = pd.DataFrame(np.zeros((X_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)], index=dfMrg.columns)

fprs, tprs, scores = [], [], []



skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)



for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    print('Fold {}\n'.format(fold))

    

    # Fitting the model

    leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])

    

    # Computing Train AUC score

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx], leaderboard_model.predict_proba(X_train[trn_idx])[:, 1])

    trn_auc_score = auc(trn_fpr, trn_tpr)

    # Computing Validation AUC score

    val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx], leaderboard_model.predict_proba(X_train[val_idx])[:, 1])

    val_auc_score = auc(val_fpr, val_tpr)  

      

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)

    

    # X_test probabilities

    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 0]

    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = leaderboard_model.predict_proba(X_test)[:, 1]

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

        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

        tprs_interp[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))

        

    # Plotting ROC for random guessing

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    

    mean_tpr = np.mean(tprs_interp, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    

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
class_survived = [col for col in probs.columns if col.endswith('Prob_1')]

probs['1'] = probs[class_survived].sum(axis=1) / N

probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / N

probs['pred'] = 0

pos = probs[probs['1'] >= 0.5].index

probs.loc[pos, 'pred'] = 1



y_pred = probs['pred'].astype(int)



submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = df_test['PassengerId']

submission_df['Survived'] = y_pred.values

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)