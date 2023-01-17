# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")





from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



import string

import warnings

warnings.filterwarnings('ignore')



SEED = 42

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv").set_index('PassengerId')

test_df = pd.read_csv("/kaggle/input/titanic/test.csv").set_index('PassengerId')



df = pd.concat([train_df, test_df])

# df = pd.concat([train_df, test_df])

# display(df)



def split_df(df):

    train_df = df[:891]

    test_df = df[891:]

    return (train_df, test_df.drop(['Survived'], axis=1))



train_df = split_df(df)[0]

test_df = split_df(df)[1]



submission_index = test_df.index

train_df.info()

print('\n\n')

test_df.info()

print('\n\n')
s = pd.Series([1, 3, 5, np.nan, 6, 8])

demo_df = pd.DataFrame({'a': s, 'b': [1, 3,5,6,7,8]})

display(demo_df)

display(demo_df.unstack().sort_values())

df_all_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()



# just renaming stuff here

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)



# here are all the features we did correlation analysis on

print(df_all_corr['Feature 1'].unique())



# just isolating for age

print(df_all_corr[df_all_corr['Feature 1'] == 'Fare'])



print(df[df['Sex']=='female']['Age'].median())

print(df[df['Sex']=='male']['Age'].median())
class_group = df.groupby(['Sex','Pclass']).median()['Age']

display(class_group)
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))



df
df['Embarked'] = df['Embarked'].fillna('S')

display(df[df['Embarked'].isnull() == True])

df[df['Fare'].isnull()==True]

df['Fare'] = df['Fare'].fillna(df.groupby(['Pclass', 'SibSp', 'Parch']).Fare.median()[3][0][0])



#check how many missing values we have left

df.isnull().sum()
# print(df[df['Cabin'].isnull() == False]['Cabin'].count())



df[df['Cabin'].isnull() == False]['Cabin'].head()



df['Deck'] = df['Cabin'].str.split('', expand = True)[1]

df['Deck'] = df['Deck'].fillna('M')

# Passenger in the T deck is changed to A

idx = df[df['Deck'] == 'T'].index

df.loc[idx, 'Deck'] = 'A'
def get_dist(): 

    counts = list(df['Deck'].value_counts())

    decks = list(df['Deck'].value_counts().index)

    deck_value_counts = dict(zip(decks, counts))





    for deck in deck_value_counts:

        deck_cond = (df['Deck'] == deck)

        surv_cond = df['Survived'] == True

        #deck_value_counts[deck] -> # ppl in the deck

        #df[surv_cond][deck_cond]['Survived'].count() -> # ppl who survived in the deck

        deck_value_counts[deck] = df[surv_cond][deck_cond]['Survived'].count()/deck_value_counts[deck]

    return deck_value_counts

    

deck_value_counts = get_dist()
deck_value_counts
plt.figure(figsize = (20, 10))



bar_count = np.arange(len(deck_value_counts))

decks = deck_value_counts.keys()

heights = deck_value_counts.values()



plt.bar(x = bar_count, height = heights)

plt.xticks(bar_count, decks)    

plt.title('Survival Rates w/respect to Deck')



plt.show()
df['Deck'] = df['Deck'].replace(['A', 'B', 'C'], 'ABC')

df['Deck'] = df['Deck'].replace(['D', 'E'], 'DE')

df['Deck'] = df['Deck'].replace(['F', 'G'], 'FG')



df['Deck'].value_counts()
(train_df, test_df) = split_df(df)

# percent_survived = train_df[train_df['Survived']==True]['Survived'].count()/train_df['Survived'].count()

percent_survived = np.round(train_df['Survived'].value_counts()[0]/train_df['Survived'].count(), decimals=4)

percent_not_survived = 1-percent_survived



plt.figure(figsize=(10, 8))

sns.countplot(train_df['Survived'])

plt.xlabel('Survival')

plt.ylabel('People')

plt.xticks((0, 1), [f'Not Survived: {percent_survived}', f'Survived: {percent_not_survived}'])

plt.show()
cont_features = ['Age', 'Fare']

surv = (train_df['Survived']==1)

fig, axs = plt.subplots(ncols = 2, nrows=2, figsize=(20, 10))



for i, feature in enumerate(cont_features):

    # dist of survival in features

    sns.distplot(train_df[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])

    sns.distplot(train_df[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

   

    #dist of feature in data

    sns.distplot(train_df[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(test_df[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

cat_features = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']



fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



# start at index 1

for x, feature in enumerate(cat_features, 1): 

    plt.subplot(2, 3, x)

    sns.countplot(x=feature, hue='Survived', data=train_df)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
df['Fare'] = pd.qcut(df['Fare'], 13)
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x = 'Fare', hue = 'Survived', data = df)



plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Countplot of survival in fare feature')

plt.show()
df['Age'] = pd.qcut(df['Age'], 10)
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x = 'Age', hue = 'Survived', data = df)

plt.title('Count of survivors amongst feature age')
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(20, 7))



df['Family_size'] = df['SibSp'] + df['Parch'] + 1



sns.countplot(x = 'SibSp', data = df, hue = 'Survived', ax = axs[0])

sns.countplot(x = 'Parch', data = df, hue = 'Survived', ax = axs[1])

sns.countplot(x = 'Family_size',data = df,hue = 'Survived', ax = axs[2])
#binning

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7:'Large', 8: 'Large', 11: 'Large'}

df['Family_size'] = df['Family_size'].map(family_map)
fig, axs = plt.subplots(figsize=(20, 7))

sns.countplot(x = 'Family_size', hue = 'Survived', data = df)
df['Ticket_freq'] = df.groupby('Ticket')['Ticket'].transform('count')

df = df.drop('Ticket', axis = 1)
plt.subplots(figsize = (20, 10))

sns.countplot(x = 'Ticket_freq', hue = 'Survived', data = df)

plt.title('Ticket freq count & Survival ')

plt.show()
# df['Name'].apply(lambda x: x.split(', '))

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand = True)[0]



df['Title'].value_counts()
women = ['Miss', 'Mrs', 'Ms', 'Dona', 'Lady', 'Mme', 'Mlle', 'the Countess']

men = ['Mr']

special = ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Jonkheer', 'Don']

children = ['Master']



titles = [women, men, special, children]



df['woman_is_married'] = df['Title'].apply(lambda x: 1.0 if x == 'Mrs' else 0.0)



df['Title'] = df['Title'].replace(['Miss', 'Mrs', 'Ms', 'Dona', 'Lady', 'Mme', 'Mlle', 'the Countess'], 'Mrs/Ms/Miss')

df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Jonkheer', 'Don'], 'Noble/Clergy/Military')

plt.subplots(figsize = (10, 7))

plt.title('Titles after grouping')

sns.countplot(x = 'Title', hue = 'Survived', data = df)



plt.show()
sns.countplot(x = 'woman_is_married', hue = 'Survived', data = df)
df
non_numeric_features = ['Embarked', 'Sex', 'Title', 'Family_size', 'Age', 'Fare']



for feature in non_numeric_features: 

    df[feature] = LabelEncoder().fit_transform(df[feature])

    

df.head()
df = df.drop(['Name', 'Parch', 'SibSp', 'Cabin'], axis = 1)
cat_features = ['Pclass', 'Embarked', 'Title', 'Family_size', 'Sex', 'Deck']

encoded_features = []



for feature in cat_features: 

    encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

    n = df[feature].nunique()

    cols = [f"{feature}_{i}" for i in range(1, n+1)]

    encoded_df = pd.DataFrame(encoded_feat, columns = cols)

    encoded_df.index = df.index

    encoded_features.append(encoded_df)

        

df = pd.concat([df, *encoded_features], axis=1)

df = df.drop(cat_features, axis=1)

pd.set_option('display.max_columns', None)

df.info()
train_df, test_df = split_df(df)



print(type(train_df))

print(type(test_df))
x_train = StandardScaler().fit_transform(train_df.drop(['Survived'], axis = 1))

y_train = train_df['Survived'].values

x_test = StandardScaler().fit_transform(test_df)



print(f"x train shape: {x_train.shape}, type: {type(x_train)}")

print(f"y train shape: {y_train.shape}, type: {type(y_train)}")

print(f"x test shape: {x_test.shape}, type: {type(x_test)}")



x_test[[1, 2, 3]]

# cool and very good
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
probs = []

importances = pd.DataFrame(np.zeros((x_train.shape[1], N)), columns=['Fold_{}'.format(i) for i in range(1, N + 1)])

importances.index =df.columns[1:]

N = 5

oob = 0

skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

tprs, fprs, scores = [], [], []



for fold, (trn_indices, val_indices) in enumerate(skf.split(x_train, y_train), 1): 

    partial_x_train = x_train[trn_indices]

    partial_y_train = y_train[trn_indices]

    x_val = x_train[val_indices]

    y_val = y_train[val_indices]

    

    #fit model and validate with x_val.

    leaderboard_model.fit(partial_x_train, partial_y_train)

    y_pred = leaderboard_model.predict_proba(x_val)[:, 1]

    

    #predict for x_test

    test_pred = leaderboard_model.predict_proba(x_test)[:, 1]

    probs.append(test_pred)

    

    #collect metrics

    fpr, tpr, thresholds = roc_curve(y_val, y_pred)

    tprs.append(tpr)

    fprs.append(fpr)

    scores.append(auc(tpr, fpr))

    

    importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_



    

    #output oob to stdout

    print(f"Fold {fold} OOB: {leaderboard_model.oob_score_}")





    
importances
# importances['Mean_Importance'] = importances.mean(axis=1)

# importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)



# plt.figure(figsize=(15, 20))

# sns.barplot(x='Mean_Importance', y=importances.index, data=importances)



# plt.xlabel('')

# plt.tick_params(axis='x', labelsize=15)

# plt.tick_params(axis='y', labelsize=15)

# plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)



# plt.show()
probs_df = pd.DataFrame(np.transpose(probs), columns = [f"Fold {n}" for n in range(1, 6)])



probs_df['avg'] = probs_df.sum(axis=1)/N

probs_df
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
predictions = list(map(lambda x: 0 if x<.5 else 1, probs_df['avg'].values))



p_df = pd.DataFrame({'Survived': predictions})

p_df.index= submission_index



display(p_df)



p_df.to_csv(r'rf_submission.csv')

from IPython.display import FileLink

FileLink(r'rf_submission.csv')
