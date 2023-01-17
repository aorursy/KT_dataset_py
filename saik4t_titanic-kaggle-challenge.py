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

train_data.head()




test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
import math

def substrings_in_string(big_string, substrings):

    #print(type(big_string))

    if type(big_string) == float:

        return np.nan

    #print(big_string, substrings)

    for substring in substrings:

        #print(substring)

        if str.find(big_string, substring) != -1:

            

            return substring

    #print big_string

    return np.nan
#Turning cabin number into Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_data['Deck']=train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
train_data["Deck"].fillna("Unknown",inplace = True)

train_data
test_data['Deck']=train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
test_data["Deck"].fillna("Unknown",inplace = True)

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

train_data['Title']=train_data['Name'].map(lambda x: substrings_in_string(x, title_list))

 

#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

train_data['Title']=train_data.apply(replace_titles, axis=1)
test_data['Title']=test_data['Name'].map(lambda x: substrings_in_string(x, title_list))

test_data['Title']=test_data.apply(replace_titles, axis=1)
test_data
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Fare","Title"]

X = pd.get_dummies(train_data[features])

X.fillna(15,inplace = True)

X.shape

X_test = pd.get_dummies(test_data[features])

X_test.fillna(15,inplace = True)

X_test.shape
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



#features = ["Pclass", "Sex", "SibSp", "Parch","Embarked", "Age", "Fare", "Cabin"]

#X = pd.get_dummies(train_data[features])



#X_test = pd.get_dummies(test_data[features])

#X_test.fillna(15,inplace = True)



model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
importances = pd.DataFrame({'feature':X.columns,'importance':np.round(model.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
X_test.shape