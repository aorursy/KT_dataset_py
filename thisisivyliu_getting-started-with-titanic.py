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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

#train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

#test_data.head()
def extract_data(df):

    update_name(df)

    return df



def update_name(df):

    df['Title'] = df['Name'].map(lambda x: extract_title(x))

    titleList = df['Title'].unique().tolist()

    print(titleList)

    df['Title']=df.apply(replace_title, axis=1)

    newtitleList = df['Title'].unique().tolist()

    print(newtitleList)

    

def replace_title(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Carlo']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    else:

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    

def extract_title(name_string):

    if (name_string.find(' ') != -1):

        return name_string.split(' ')[1][:-1]

    return np.nan



train_data = extract_data(train_data)

test_data = extract_data(test_data)

#print(train_data.to_string())

#print(test_data.to_string())
'''

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)'''
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Title"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

#print(X)

print(X_test)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")