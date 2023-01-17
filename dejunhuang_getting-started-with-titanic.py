# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
def concat_df(train_df, test_df):

    return pd.concat([train_df, test_df], sort=True).reset_index(drop=True)



def split_df(df_all):

    return df_all.iloc[:891], df_all.iloc[891:].drop('Survived', axis=1)
df_all = concat_df(train_data, test_data)
df_all.info()
df_all.hist(figsize=(20,12))

plt.show()
df_all_corr = df_all.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()

df_all_corr[df_all_corr['level_0']=='Age']
df_all.groupby(['Pclass','Sex']).median()['Age']
df_all['Age'] = df_all.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all['Family'] = df_all['Parch'] + df_all['SibSp']
df_all_corr = df_all.corr().unstack().sort_values(kind='quicksort', ascending=False).reset_index()

df_all_corr[df_all_corr['level_0']=='Survived']
ticket_counts = df_all['Ticket'].value_counts()



for ticket, count in ticket_counts.iteritems():

    df_all.loc[df_all['Ticket'] == ticket, 'Ticket'] = count



df_all['Ticket'] = df_all['Ticket'].astype('int')
df_all_corr = df_all.corr().unstack().sort_values(kind='quicksort', ascending=False).reset_index()

df_all_corr[df_all_corr['level_0']=='Survived']
df_all.head()
df_all.groupby('Embarked')['Survived'].sum()

df_all['Embarked'].fillna(value=df_all['Embarked'].mode().loc[0], inplace=True)
dict_embarked = {'S': 217,

                 'C': 93,

                 'Q': 30}



df_all.replace({'Embarked': dict_embarked}, inplace=True)
train_df, test_df = split_df(df_all)
test_df['Fare'].fillna(value=test_df['Fare'].median(), inplace=True)
y = train_df["Survived"].astype('int')



features = ['Age', 'Fare', 'Embarked', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Family', 'Ticket']



X = pd.get_dummies(train_df[features])

X_test = pd.get_dummies(test_df[features])
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#record keeping for comparison later with old methods

output_adv_age = output['Survived'].copy()



X_adv_age = X['Age'].copy()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data['Age'].fillna(train_data['Age'].mode().loc[0], inplace=True)

test_data['Age'].fillna(train_data['Age'].mode().loc[0], inplace=True)

test_data['Fare'].fillna(train_data['Fare'].mode().loc[0], inplace=True)



y = train_data["Survived"]



features = ['Age', 'Fare', 'Embarked', 'Pclass', 'Sex', 'SibSp', 'Parch']



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")



output_normal_age = output['Survived'].copy()



X_normal_age = X['Age'].copy()



((X_adv_age == X_normal_age)*1).sum()
((output_adv_age == output_normal_age)*1).sum()/len(output_adv_age)