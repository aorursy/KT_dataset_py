# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

%matplotlib inline

import warnings 

warnings.simplefilter('ignore')

from scipy import stats

from scipy.stats import norm 
sns.set(rc = {'figure.figsize': (12,8)})
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
print(train.shape)

print(test.shape)
train.info()
test.info()
fig,(ax1,ax2) = plt.subplots(2,1 , figsize=(16,9))

sns.heatmap(train.isnull(), yticklabels = False , ax = ax1)

sns.heatmap(test.isnull(), yticklabels = False ,  ax = ax2)

plt.show()
print('Train')

for i in ['Age','Cabin']:

    print(f'Missed {i} (all) ', round(train[i].isnull().sum()/train.shape[0] *100,2) ,'%')

    print(f'Missed Age(male) ', round(train[train['Sex']=='male'][i].isnull().sum()/train.shape[0] *100,2) ,'%')

    print(f'Missed Age(female) ', round(train[train['Sex']=='female'][i].isnull().sum()/train.shape[0] *100,2) ,'%')

    print('*'*50)
print('Test')

for i in ['Age','Cabin','Fare']:

    print(f'Missed  {i} ', round(test[i].isnull().sum()/test.shape[0] *100,2) ,'%')

    print(f'Missed Age(male) ', round(test[train['Sex']=='male'][i].isnull().sum()/test.shape[0] *100,2) ,'%')

    print(f'Missed Age(female) ', round(test[test['Sex']=='female'][i].isnull().sum()/test.shape[0] *100,2) ,'%')

    print('*'*50)
for i in train['Sex'].unique():

    count_male_female = (train.loc[(train['Sex'] == i) ]['Sex'].count())

    Survived = np.round((train.loc[(train['Sex'] == i) & (train['Survived'] == 1)]['Sex'].count() / count_male_female) * 100, 2)

    print(f'All {i}: {count_male_female}')

    print(f'Survived {i}: {Survived} %')

    print(f'Not Survived {i}: {100 - Survived:.2f} %')

    print('*'*50)
sns.catplot(x="Pclass", y="Age",

                hue="Survived", col="Sex",

                data=train, kind="swarm",

                height=5);
sns.catplot(x="Pclass", y="Fare",

                hue="Survived", col="Sex",

                data=train, kind="swarm",

                height=5);
df = pd.concat((train,test))
df[df['Fare'].isnull()]
median_fare = df.query('(Pclass == 3) & (Embarked == "S")')['Fare'].median()

print(f'median_fare: {median_fare}')
df['Fare'].fillna(median_fare, inplace = True)
df.describe()['Fare']
bins = [-np.inf, 1,20,50, np.inf]

label = ['Crew','Passenger_1','Passenger_2','Passenger_3']

df['Fare_category'] = pd.cut(df['Fare'], bins, labels = label)
df['Fare_category'].value_counts()
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16,5))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)



bar_1 = sns.distplot(df['Fare'], ax = ax1, fit = norm)

bar_1.legend([f"skew {round(df['Fare'].skew(),2)}"])



stats.probplot(df['Fare'], plot= ax2)



plt.show()


df['Fare'] = np.log(df['Fare'] +1)
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16,5))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)



bar_1 = sns.distplot(df['Fare'], ax = ax1, fit = norm)

bar_1.legend([f"skew {round(df['Fare'].skew(),2)}"])

stats.probplot(df['Fare'], plot= ax2)

ax1.set_title('Train Fare', fontsize = 14)



plt.show()
df['Name'] = df['Name'].str.lower()
all_name = df['Name'].unique()
all_name
all_name_title = pd.Series(data = all_name, index = all_name, name = 'All_Name')
def clean_name_for_title(x):

    x = re.split('[^a-z]',re.sub('[.,()]+', ' ', str(x)))

    

    if 'mr' in x:

        x = 'mr'

        

    elif 'miss' in x:

        x = 'miss'

        

    elif 'mrs' in x:

        x = 'mrs'

        

    elif 'master' in x:

        x = 'master'

        

    elif 'don' in x:

        x = 'don'

        

    elif 'rev' in x:

        x = 'rev'

        

    elif 'dr' in x:

        x = 'dr'

        

    elif 'mme' in x:

        x = 'mme'

        

    elif 'ms' in x:

        x = 'ms'

        

    elif 'major' in x:

        x = 'major'

        

    elif 'dona' in x:

        x = 'dona'

        

    else:

        x = 'other'

        

    return x
all_name_title = all_name_title.map(clean_name_for_title)
all_name_title
df['Title'] = df['Name'].map(all_name_title)
df['Title'].value_counts()


df.loc[(df['Title'] == 'other') & (df['Sex'] == 'male'),'Title'] = 'mr'

df.loc[(df['Title'] == 'other') & (df['Sex'] == 'female'),'Title'] = 'mrs'

df.loc[df['Title'] == 'ms','Title' ] = 'mrs'

df.loc[df['Title'] == 'mme','Title' ] = 'mrs'

df.loc[df['Title'] == 'dona','Title' ] = 'mrs'

df.loc[df['Title'] == 'don','Title' ] = 'mr'

df.loc[df['Title'] == 'major','Title' ] = 'mr'

df.loc[(df['Title'] == 'dr') & (df['Sex'] == 'male'),'Title'] = 'mr'

df.loc[(df['Title'] == 'dr') & (df['Sex'] == 'female'),'Title'] = 'mrs'

df.loc[df['Title'] == 'rev','Title' ] = 'mr'
pd.pivot_table(data = df, columns = 'Title', index = 'Sex',aggfunc='count',values = 'PassengerId' )
df['Title'].value_counts()
df.drop('Name', axis = 1, inplace = True)
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16,5))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)



bar_1 = sns.distplot(df['Age'], ax = ax1, fit = norm)

bar_1.legend([f"skew {round(df['Age'].skew(),2)}"])

stats.probplot(df['Age'], plot= ax2)

ax1.set_title('Train Age', fontsize = 14)



plt.show()
median_age_male = df.loc[df['Sex'] == 'male']['Age'].median()

print(f'Median age male : {median_age_male}')

df.loc[(df['Sex'] == 'male') & (df['Age'].isnull()), 'Age'] = median_age_male
median_age_female = df.loc[df['Sex'] == 'female']['Age'].median()

print(f'Median age female : {median_age_female}')

df.loc[(df['Sex'] == 'female') & (df['Age'].isnull()), 'Age'] = median_age_female
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16,5))

fig.subplots_adjust(hspace = 0.3, wspace = 0.3)



bar_1 = sns.distplot(df['Age'], ax = ax1, fit = norm)

bar_1.legend([f"skew {round(df['Age'].skew(),2)}"])

stats.probplot(df['Age'], plot= ax2)

ax1.set_title('Train Age', fontsize = 14)



plt.show()
bins = [-np.inf,10,22,55,np.inf]

label = ['Child','Teenager','Adult','Old']

df['Type_People'] = pd.cut(df['Age'], bins, labels = label)
def make_conj(df, feature1, feature2):

    df[feature1 + '_' + feature2] = df[feature1].astype(str).str[:1] + '_' + df[feature2].astype(str)

    

# male + Adult  = m_Adult

# female + Child = f_Child

# ....
make_conj(df, 'Sex', 'Type_People')
df['Sex_Type_People'].value_counts()
sns.catplot(x="Pclass", y="Fare",

                hue="Survived", col="Sex_Type_People",

                data=df, kind="swarm",

                height=5,col_wrap=4);
df['Deck'] = df['Cabin'].str[:1]
df['Deck'].value_counts(dropna = False)
pd.pivot_table(data = df, columns = 'Deck', index = 'Pclass',aggfunc='count',values = 'PassengerId' )
for category in df['Sex_Type_People'].unique():

    try:

        mode_deck_1 = df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 1)]['Deck'].mode()[0]

        mode_deck_2 = df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 2)]['Deck'].mode()[0]

        mode_deck_3 = df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 3)]['Deck'].mode()[0]



        df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 1) & (df['Deck'].isnull()), 'Deck'] = mode_deck_1

        df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 2) & (df['Deck'].isnull()), 'Deck'] = mode_deck_2

        df.loc[(df['Sex_Type_People'] == category) & (df['Pclass'] == 3) & (df['Deck'].isnull()), 'Deck'] = mode_deck_3

    finally:

        mode_deck_1 = df.loc[df['Pclass'] == 1]['Deck'].mode()[0]

        mode_deck_2 = df.loc[df['Pclass'] == 2]['Deck'].mode()[0]

        mode_deck_3 = df.loc[df['Pclass'] == 3]['Deck'].mode()[0]



        df.loc[(df['Pclass'] == 1) & (df['Deck'].isnull()), 'Deck'] = mode_deck_1

        df.loc[(df['Pclass'] == 2) & (df['Deck'].isnull()), 'Deck'] = mode_deck_2

        df.loc[(df['Pclass'] == 3) & (df['Deck'].isnull()), 'Deck'] = mode_deck_3
df.loc[df['Deck'] == 'E', 'Pclass'] = 1

df.loc[(df['Pclass'] == 1) &(df['Deck'] == 'T'), 'Deck'] = 'C'

df.loc[(df['Pclass'] == 3) &(df['Deck'] == 'G'), 'Deck'] = 'F'
df['Deck'].value_counts(dropna = False)
sns.catplot(x="Pclass", y="Age",

                hue="Survived", col="Deck",

                data=df, kind="swarm",

                height=5,col_wrap=4);
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)
df['Embarked'].unique()
df.info()
for i in ['Pclass', 'Sex', 'SibSp', 'Parch',

       'Ticket', 'Cabin', 'Embarked', 'Fare_category', 'Title',

       'Sex_Type_People', 'Deck']:

    df[i] = df[i].astype('object')
df.drop(['Cabin', 'Ticket'], axis = 1, inplace = True)
columns = [

            'PassengerId',

           'Survived',

           'Pclass',

           #'Sex',

           #'Age',

           #'SibSp',

           #'Parch',

           'Fare',

           #'Embarked',

           'Fare_category',

           #'Title',

           #'Type_People',

           'Sex_Type_People',

            #'Deck'

]
df_test = df[columns].copy()
df_test = pd.get_dummies(df_test)
df_test.head()
df_test.shape
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
train = df_test[:len(train)]

test = df_test[len(train):]
train.head()
X_train = train.drop(['PassengerId','Survived'],axis = 1)

y_train = train['Survived']

X_t_target = test.drop(['PassengerId','Survived'],axis = 1)
X_train['Fare'] = (X_train['Fare'] - X_train['Fare'].mean())/X_train['Fare'].std()
X_t_target['Fare'] = (X_t_target['Fare'] - X_t_target['Fare'].mean())/X_t_target['Fare'].std()
X_train.head()
def score_model(model, X_train = X_train, y_train = y_train ):

    

    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    

    model.fit(X_train,y_train)

    

    score_train = model.score(X_train,y_train)

    

    score_val = cross_val_score(model, X_train, y_train, cv = cv, n_jobs = -1).mean()

    

    return f'Train data: {score_train} ****** cross_val_score: {score_val}'
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

KNN = KNeighborsClassifier()
KNN
score_model(KNN)
params = {

        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],

         'n_neighbors':np.arange(1,20),

         'p':np.arange(1,3),

         'weights':['uniform', 'distance']

                    }
gsc = GridSearchCV(KNN, params, n_jobs = -1, cv = 5)
gsc.fit(X_train, y_train) 
gsc.best_params_
KNN_gsc = gsc.best_estimator_
score_model(KNN_gsc)
KNN_gsc.fit(X_train, y_train)
y_pred = KNN_gsc.predict(X_t_target)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],

                       'Survived':y_pred.astype(int)})
submission.to_csv('submission.csv', index = False)