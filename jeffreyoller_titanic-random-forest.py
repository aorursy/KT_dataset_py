import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
weights_df = pd.read_csv('/kaggle/input/average-weights-by-age/Average_weights.csv')
age_mean =  np.mean(df.Age.append(test_df.Age))

df.loc[(df.Age.isna()),'Age'] = age_mean

test_df.loc[(test_df.Age.isna()),'Age'] = age_mean
df = df.merge(weights_df,left_on=['Age','Sex'],right_on = ['Age','Sex'],how='left')

df.loc[(df.Weight.isna()) & (df.Sex == 'male'),'Weight'] = 155

df.loc[(df.Weight.isna()) & (df.Sex == 'female'),'Weight'] = 128



test_df = test_df.merge(weights_df,left_on=['Age','Sex'],right_on = ['Age','Sex'],how='left')

test_df.loc[(test_df.Weight.isna()) & (test_df.Sex == 'male'),'Weight'] = 155

test_df.loc[(test_df.Weight.isna()) & (test_df.Sex == 'female'),'Weight'] = 128
test_df['Carriability'] = 1-(test_df.Weight - test_df.Weight.min())/(test_df.Weight.max()-test_df.Weight.min())

df['Carriability'] = 1-(df.Weight - df.Weight.min())/(df.Weight.max()-df.Weight.min())
df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

test_df['Title'] = test_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

title_dict = {

    "Capt": "Military",

    "Col": "Military",

    "Major": "Military",

    "Jonkheer": "Noble",

    "Don": "Noble",

    "Dona": "Noble",

    "Sir" : "Noble",

    "Dr": "Professional",

    "Rev": "Professional",

    "the Countess":"Noble",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Noble"

}

df.Title = df.Title.map(title_dict)

test_df.Title = test_df.Title.map(title_dict)

df.Title.unique()
y=df.Survived

features = ['Pclass','Sex','SibSp','Parch','Embarked','Carriability','Title']

X = pd.get_dummies(df[features])

X_test = pd.get_dummies(test_df[features])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200,max_depth=6,random_state=1)

model.fit(X,y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerID':test_df.PassengerId,'Survived':predictions})

output.to_csv('my_submission_2.csv', index = False)

print('Your submission was successfully saved!')
output = pd.DataFrame({'PassengerID':test_df.PassengerId,'Survived':predictions})

output.to_csv('my_submission_2.csv', index = False)

print('Your submission was successfully saved!')
features = ['Survived','Pclass','Sex','SibSp','Parch','Embarked','Carriability','Title']



results = pd.get_dummies(df[features]).corr()

feature_list = results.columns[1:]

results
import matplotlib.pyplot as plt



%matplotlib inline



importances = list(model.feature_importances_)



plt.style.use('seaborn-darkgrid')



x_values = list(range(len(importances)))



plt.bar(x_values, importances, orientation = 'vertical')



plt.xticks(x_values, feature_list, rotation='vertical')



plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')