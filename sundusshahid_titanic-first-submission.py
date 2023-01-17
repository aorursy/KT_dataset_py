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
%matplotlib inline



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



# Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split





# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')
# Import train & test data 

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv') # example of what a submission should look like



# View the training data

train.head(15)
train.Age.plot.hist()
test.head()
gender_submission.head()
train.describe()
# Plot graphic of missing values

missingno.matrix(train, figsize = (30,10))
train.isnull().sum()
df_bin=pd.DataFrame()

df_con=pd.DataFrame()

train.dtypes
train.head()
# How many people survived?

fig = plt.figure(figsize=(20,1))

sns.countplot(y='Survived', data=train);

print(train.Survived.value_counts())
df_bin['Survived']=train['Survived']

df_con['Survived']=train['Survived']
df_bin.head()
df_con.head()
sns.distplot(train.Pclass)
train.Pclass.isnull().sum()
df_bin['Pclass']=train['Pclass']

df_con['Pclass']=train['Pclass']
train.Name.value_counts()
data = [train, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)
train.Title.value_counts()

df_bin['Title']=train['Title']

df_con['Title']=train['Title']
train.head()
plt.figure(figsize=(20, 5))

sns.countplot(y="Sex", data=train);
train.Sex.isnull().sum()
train.Sex.head()
df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female



df_con['Sex'] = train['Sex']
# How does the Sex variable look compared to Survival?

# We can see this because they're both binarys.

fig = plt.figure(figsize=(10, 10))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'});



# How many missing values does age have?

train.Age.isnull().sum()
#train_df=train

#test_df=test

data = [train, test]



for dataset in data:

    mean = train["Age"].mean()

    std = test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train["Age"].astype(int)

train["Age"].isnull().sum()
#Once the Age values have been fixed up, we can add them to our sub dataframes.

df_bin['Age'] = pd.cut(train['Age'], 10) # bucketed/binned into different categories

df_con['Age'] = train['Age'] # non-bucketed
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    """

    Function to plot counts and distributions of a label variable and 

    target variable side by side.

    ::param_data:: = target dataframe

    ::param_bin_df:: = binned dataframe for countplot

    ::param_label_column:: = binary labelled column

    ::param_target_column:: = column you want to view counts and distributions

    ::param_figsize:: = size of figure (width, height)

    ::param_use_bin_df:: = whether or not to use the bin_df, default False

    """

    if use_bin_df: 

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});
# How many missing values does SibSp have?

train.SibSp.isnull().sum()
# What values are there?

train.SibSp.value_counts()
# Add SibSp to subset dataframes

df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
# Visualise the counts of SibSp and the distribution of the values

# against Survived

plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(20, 10))
# How many missing values does Parch have?

train.Parch.isnull().sum()
# What values are there?

train.Parch.value_counts()
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
# Visualise the counts of Parch and the distribution of the values

# against Survived

#plot_count_dist(train, 

#                bin_df=df_bin,

#                label_column='Survived', 

 #               target_column='Parch', 

 #               figsize=(20, 10))
train.head()
df_con.head()
# How many missing values does Ticket have?

train.Ticket.isnull().sum()
# How many kinds of ticket are there?

sns.countplot(y="Ticket", data=train);
# How many kinds of ticket are there?

train.Ticket.value_counts()
# How many unique kinds of Ticket are there?

print("There are {} unique Ticket values.".format(len(train.Ticket.unique())))
# How many missing values does Fare have?

train.Fare.isnull().sum()
# How many different values of Fare are there?

sns.countplot(y="Fare", data=train);
# How many unique kinds of Fare are there?

print("There are {} unique Fare values.".format(len(train.Fare.unique())))
#Add Fare to sub dataframes

df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5) # discretised
# What do our Fare bins look like?

df_bin.Fare.value_counts()
# Visualise the Fare bin counts as well as the Fare distribution versus Survived.

plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20,10), 

                use_bin_df=True)
# How many missing values does Cabin have?

train.Cabin.isnull().sum()
# What do the Cabin values look like?

train.Cabin.value_counts()
train.Embarked.isnull().sum()
train.Embarked.value_counts()
# What do the counts look like?

sns.countplot(y='Embarked', data=train);
# Add Embarked to sub dataframes

df_bin['Embarked'] = train['Embarked']

df_con['Embarked'] = train['Embarked']
# Remove Embarked rows which are missing values

print(len(df_con))

df_con = df_con.dropna(subset=['Embarked'])

df_bin = df_bin.dropna(subset=['Embarked'])

print(len(df_con))

print(len(df_bin))
df_bin.head()
# One-hot encode binned variables

one_hot_cols = df_bin.columns.tolist()

one_hot_cols.remove('Survived')

df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)



df_bin_enc.head()
df_con.head()
df_con_enc=df_con.apply(LabelEncoder().fit_transform)

df_con_enc.head()
# Let's look at df_con_enc

df_con_enc.head(20)


# Seclect the dataframe we want to use first for predictions

selected_df = df_con_enc
selected_df.head()
# Split the dataframe into data and labels

X_train = selected_df.drop('Survived', axis=1) # data

y_train = selected_df.Survived # labels
X_train.head()
y_train.shape
#Random forest tree classifier

rfc=RandomForestClassifier()
param={

    'n_estimators':list(range(100,501)),

    "max_depth":[2,3,4,5,6,7,8,9,10]

}
rscv=RandomizedSearchCV(rfc, param_distributions =param, n_jobs= -1, verbose=10, n_iter=10, 

                        cv=10, scoring = "accuracy")
rscv.fit(X_train,y_train)
rscv.best_estimator_
rfc=RandomForestClassifier(max_depth=9, n_estimators=275)
cross_val_score(rfc, X_train, y_train, cv=10, scoring="accuracy").mean()
rfc.fit(X_train, y_train)
# We need our test dataframe to look like this one

X_train.head()
# Our test dataframe has some columns our model hasn't been trained on

test.head()
# Let's look at test, it should have one hot encoded columns now

test.head()
# Create a list of columns to be used for the predictions

wanted_test_columns = X_train.columns

wanted_test_columns
# Make a prediction using the random forest model on the wanted columns

predictions = rfc.predict(test[wanted_test_columns].apply(LabelEncoder().fit_transform))
predictions
# Our predictions array is comprised of 0's and 1's (Survived or Did Not Survive)

predictions[:20]
# Create a submisison dataframe and append the relevant columns

submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = predictions # our model predictions on the test dataset

submission.head()
# What does our submission have to look like?

gender_submission.head()
# Let's convert our submission dataframe 'Survived' column to ints

submission['Survived'] = submission['Survived'].astype(int)

print('Converted Survived column to integers.')
# How does our submission dataframe look?

submission.head()
# Are our test and submission dataframes the same length?

if len(submission) == len(test):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
# Convert submisison dataframe to csv for submission to csv 

# for Kaggle submisison

submission.to_csv('sub9.csv', index=False)



print('Submission CSV is ready!')
# Check the submission csv to make sure it's in the right format

submissions_check = pd.read_csv("sub9.csv")

submissions_check.head(8)
train.head(25)