#Imports





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import mode
#read and review data



df = pd.read_csv('../input/train.csv')

df.head(10)
#yeah, let's not

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
#well, what do we have heeeeereee........

df.info()
#guessing games



#If I were trying harder I'd RNG some numbers in based on the distribution of the known data



age_median = df['Age'].median()

df['Age'] = df['Age'].fillna(age_median)



#some dumb hacks to get the mode because str don't work anymore

embarkedonly = pd.DataFrame(df['Embarked'])

embarkedonly = embarkedonly.dropna()

embarkedonly = embarkedonly['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

mode_embarked = mode(embarkedonly)[0][0]

print(mode_embarked) #it's 2





df['Embarked'] = df['Embarked'].fillna('S')

print(df.info())
#words are hard, let's use numbers

df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)

df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()

cols = [cols[1]] + cols[0:1] + cols[2:]

df = df[cols]



df.head(10)
train_data = df.values
#forests.



model = RandomForestClassifier(n_estimators = 100)

model = model.fit(train_data[0:,2:], train_data[0:,0])
df_test = pd.read_csv('../input/test.csv')

df_test.head(10)
#same shit different data



df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)



df_test['Age'] = df_test['Age'].fillna(age_median)

fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')

df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:

                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])

                            else x['Fare'], axis=1)



df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})

df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})



df_test = df_test.drop(['Sex', 'Embarked'], axis=1)



test_data = df_test.values
output = model.predict(test_data[:,1:])



result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])



df_result.head(10)
result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])



df_result.to_csv('titanic_1-1.csv', index=False)