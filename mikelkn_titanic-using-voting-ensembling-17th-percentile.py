import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('The size of the training set: ', train.shape)

print('The size of the test set is: ' ,test.shape)
test.head()
train.describe()
# To get more information about the data you are working with, it is good to use the info() method



train.info()
Passenger_Id = test['PassengerId']
#we keep this for when we will be separating DATA back into train and test 



train_rowsize = train.shape[0]

test_rowsize = test.shape[0]
#The different data types 

train.dtypes.value_counts()
test.dtypes.value_counts()
data = pd.concat((train, test))



# we drop the preditor from the data 

data.drop('Survived', axis = 1, inplace = True)

data.drop('PassengerId', axis = 1, inplace = True)
data.hist(figsize=(10,10));
#we find out the percentage of survivers by gender

data.Sex.value_counts(normalize = True)
embarked_counts = data.Embarked.value_counts(normalize = True)

embarked_counts
embarked_counts.plot(kind='bar')

plt.title("Passengers per boarding gates");
train['Died'] = 1 - train['Survived']

train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind = 'bar', figsize = (10, 5), stacked = True);
ax = plt.subplot()

ax.set_ylabel('Average fare')

data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(10, 5), ax = ax);
plt.figure(figsize=(10, 5))

plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 

         stacked=True,bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();
#Here is a list of all the features with Nans and the number of null for each features

null_values = data.columns[data.isnull().any()]

null_features = data[null_values].isnull().sum().sort_values(ascending = False)

missing_data = pd.DataFrame({'No of Nulls' :null_features})

missing_data
test.isnull().sum()
train.isnull().sum()
#Fare, embarked, Cabin and Age all have missing values. Lets plot the missing values

import warnings

warnings.filterwarnings('ignore')            #to silence warnings



%matplotlib inline

sns.set_context('talk')

sns.set_style('ticks')

sns.set_palette('dark')



plt.figure(figsize= (10, 5))

plt.xticks(rotation='90')

ax = plt.axes()

sns.barplot(null_features.index, null_features)

ax.set(xlabel = 'Features', ylabel = 'Number of missing values', title = 'Features with Missing values');
data["Embarked"] = data["Embarked"].fillna('S')
data[data['Fare'].isnull()]
def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

#'S'

       #print(median_fare)

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



data=fill_missing_fare(data)
%matplotlib inline



sns.set_context('notebook')

sns.set_style('ticks')

sns.set_palette('dark')



plt.figure(figsize= (10, 5))



ax = sns.distplot(data["Age"].dropna(),   #plot only the numerical data

                  color = 'green',

                 kde = False)    

ax.grid(True)



ax.set(xlabel = 'Age', ylabel = 'Number of people', title = 'Age range of Passengers');
# we will genrate a set of random values from 0 to 80 and missing ages with any one of these values



sizeof_null = data["Age"].isnull().sum()

rand_age = np.random.randint(0, 80, size = sizeof_null)
 # fill NaN values in Age column with random values generated

    

age_slice = data["Age"].copy()

age_slice[np.isnan(age_slice)] = rand_age

data["Age"] = age_slice

data["Age"] = data["Age"].astype(int)
data['Age'] = data['Age'].astype(int)

data.loc[ data['Age'] <= 18, 'Age'] = 0

data.loc[(data['Age'] > 18) & (data['Age'] <= 35), 'Age'] = 1

data.loc[(data['Age'] > 35) & (data['Age'] <= 60), 'Age'] = 2

data.loc[(data['Age'] > 60) & (data['Age'] <= 80), 'Age'] = 3



data['Age'].value_counts()
data.sample(10)
data['Cabin'].dropna().sample(10)
data["Deck"]=data['Cabin'].str[0]



data['Deck'].unique()
data['Deck'] = data['Deck'].fillna('H')   # replacing the nan with H
data['Deck'].unique()
# we include a new feauture, the Familysize including the passengers

data["FamilySize"] = data["SibSp"] + data["Parch"]+ 1

data['FamilySize'].value_counts()
data.loc[ data['FamilySize'] == 1, 'FSize'] = 'Single family'

data.loc[(data['FamilySize'] > 1) & (data['FamilySize'] <= 5), 'FSize'] = 'Small Family'

data.loc[(data['FamilySize'] > 5), 'FSize'] = ' Extended Family'
data.head()
le = LabelEncoder()



data['Sex'] = le.fit_transform(data['Sex'])

data['Embarked'] = le.fit_transform(data['Embarked'])

data['Deck'] = le.fit_transform(data['Deck'])

data['FSize'] = le.fit_transform(data['FSize'])
data['Sex'].unique()
data.dtypes.value_counts()
data = data.drop(['Name', 'Ticket','Cabin',], axis = 1)
#we check for skewness in  data



skew_limit = 0.75

skew_vals = data.skew()



skew_cols = (skew_vals

             .sort_values(ascending=False)

             .to_frame()

             .rename(columns={0:'Skewness'})

            .query('abs(Skewness) > {0}'.format(skew_limit)))



skew_cols
print("There are {} skewed numerical features to  transform".format(skew_cols.shape[0]))
tester = 'Deck'

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16,5))

#before normalisation

data[tester].hist(ax = ax_before)

ax_before.set(title = 'Before nplog1p', ylabel = 'Frequency', xlabel = 'Value')



#After normalisation

data[tester].apply(np.log1p).hist(ax = ax_after)

ax_after.set(title = 'After nplog1p', ylabel = 'Frequency', xlabel = 'Value')



fig.suptitle('Field "{}"'.format(tester));
skewed = skew_cols.index.tolist()

data[skewed] = data[skewed].apply(np.log1p)
# Correlation between the features and the predictor- Survived

predictor = train['Survived']

correlations = data.corrwith(predictor)

correlations = correlations.sort_values(ascending = False)

# correlations

corrs = (correlations

            .to_frame()

            .reset_index()

            .rename(columns={'level_0':'feature1',

                                0:'Correlations'}))

corrs
plt.figure(figsize= (10, 5))

ax = correlations.plot(kind = 'bar')

ax.set(ylabel = 'Pearson Correlation', ylim = [-0.4, 0.4]);
#importing libraries

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import VotingClassifier

# First i need to break up my data into train and test

train_new = data[:train_rowsize]

test_new = data[train_rowsize:]

test_new.shape
train_new.dtypes.value_counts()
train_new.head()
test_new.dtypes.value_counts()
test_new.head()
n_folds = 5



kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_new)

y_train = train.Survived

n_folds = 5

    

def f1_score (model): 

    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_new)

    rmse = np.sqrt(cross_val_score(model, train_new, y_train, scoring = 'f1', cv = kf))

    # f1 because it is the sweet spot between recall and precision

    return (rmse)
logreg = LogisticRegression()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()

xgb = XGBClassifier()

lgbm = LGBMClassifier()
# from sklearn.metrics import SCORERS

# print(SCORERS.keys())
score = f1_score(logreg)

print("\nLogistic regression score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
score = f1_score(rf)

print("\nRandom Forest score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
score = f1_score(gboost)

print("\nGradient Boosting Classifier score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
score = f1_score(xgb)

print("\neXtreme Gradient BOOSTing score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
score = f1_score(lgbm)

print("\nLight Gradient Boosting score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
all_classifier = VotingClassifier(estimators=[('logreg', logreg), ('rf', rf), 

                                              ('gboost', gboost), ('xgb', xgb),

                                             ('lgbm', lgbm)], voting='soft')



VC = all_classifier.fit(train_new, y_train)
score = f1_score(VC)

print("\nVoting Classifier score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))
prediction = VC.predict(test_new)
titanic_submission = pd.DataFrame ({"PassengerId": test["PassengerId"],

                             "Survived": prediction})

titanic_submission.to_csv('Titanic_Submission.csv', index = False)



titanic_submission.sample(10)
