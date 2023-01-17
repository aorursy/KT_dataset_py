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
# get training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# get test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# Describe training data

train_data.describe()
# Describe test data

test_data.describe()
# Look null values for train data

train_data.isnull().sum()
# Look null values for test data

test_data.isnull().sum()
test_data[test_data["Fare"].isnull()]
test_data.loc[152,'Fare'] = 20.2125
test_data.loc[152]
test_data.isnull().sum()
def find_titles(df):

    titles=set()

    for name in df:

        if name.find('.'):

            title = name.split('.')[0].split()[-1]

            titles.add(title)

    return titles
titles = find_titles(train_data["Name"]).union(find_titles(test_data["Name"]))
titles
my_titles_dict = {"Mr": ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Mr', 'Rev', 'Sir'],

                 "Mrs": ['Countess', 'Dona', 'Lady', 'Mme', 'Mrs'],

                 "Miss": ['Miss', 'Mlle','Ms'],

                 "Master": ['Master'],

                 "Other": ['Dr']}
def set_titles(df):

    if "Title" in df.columns:

        print("Title column already exists")

    else:

        pd.DataFrame.insert(df, len(df.columns),"Title","",False)

    

    for i in range(len(df)):

        name=df.loc[i,'Name']

        if name.find('.'):

            title = name.split('.')[0].split()[-1]

            for key in my_titles_dict:

                if title in my_titles_dict[key]:

                    if key == "Other":

                        if df.loc[i,'Sex']=='female':

                            df.loc[i, 'Title'] = 'Mrs'

                        else:

                            df.loc[i, 'Title'] = 'Mr'

                    else:        

                        df.loc[i, 'Title'] = key            

        i+=1

    return df
train_data=set_titles(train_data)

test_data=set_titles(test_data)
test_data.tail(5)
train_master=train_data[['Master' in x for x in train_data['Title']]][train_data["Age"].notnull()]

test_master=test_data[['Master' in x for x in test_data['Title']]][test_data["Age"].notnull()]

train_mr=train_data[['Mr' in x for x in train_data['Title']]][train_data["Age"].notnull()]

test_mr=test_data[['Mr' in x for x in test_data['Title']]][test_data["Age"].notnull()]

train_mrs=train_data[['Mrs' in x for x in train_data['Title']]][train_data["Age"].notnull()]

test_mrs=test_data[['Mrs' in x for x in test_data['Title']]][test_data["Age"].notnull()]

train_miss=train_data[['Miss' in x for x in train_data['Title']]][train_data["Age"].notnull()]

test_miss=test_data[['Miss' in x for x in test_data['Title']]][test_data["Age"].notnull()]
train_master_mean = train_master["Age"].mean()

test_master_mean = test_master["Age"].mean()

train_mr_mean = train_mr["Age"].mean()

test_mr_mean = test_mr["Age"].mean()

train_mrs_mean = train_mrs["Age"].mean()

test_mrs_mean = test_mrs["Age"].mean()

train_miss_mean = train_miss["Age"].mean()

test_miss_mean = test_miss["Age"].mean()
print(train_master_mean)

print(test_master_mean)

print(train_mr_mean)

print(test_mr_mean)

print(train_mrs_mean)

print(test_mrs_mean)

print(train_miss_mean)

print(test_miss_mean)
train_mean_values_dict = {"Mr": 33.62,

                 "Mrs": 35.99,

                 "Miss": 21.84,

                 "Master":4.57}

test_mean_values_dict = {"Mr": 33.98,

                 "Mrs": 38.90,

                 "Miss": 21.77,

                 "Master":7.40}
def impute_to_age (df1, df2):

    for i in range(len(df1)):

        title=df1.loc[i,'Title']

        age=df1.at[i,'Age'].astype(float)

        if np.isnan(age):

            df1.loc[i,'Age']=train_mean_values_dict.get(title)

        i+=1

    for j in range(len(df2)):

        title=df2.at[j,'Title']

        age=df2.at[j,'Age'].astype(float)

        if np.isnan(age):

            df2.loc[j,'Age']=test_mean_values_dict.get(title)

        j+=1
impute_to_age(train_data, test_data)
train_data.isnull().sum()
test_data.isnull().sum()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X.drop(['Sex_male'], axis=1, inplace=True)

X_test.drop(['Sex_male'], axis=1, inplace=True)



model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
X_test