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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_data.head()
train_data.describe()
# importing the visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



#copying data to manipulate

df = train_data.copy()
plt.style.use('seaborn')

fig, axs = plt.subplots(ncols=3,figsize=(15,5))



sns.countplot(x='Survived',data=df, ax=axs[0])

sns.countplot(x='Sex',hue='Survived',data=df, ax=axs[1])

sns.countplot(x='Sex',data=df,ax=axs[2])



plt.show()
fig = plt.figure()

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)





slices_1 = [df['Survived'].sum(),(len(df)-df['Survived'].sum())]

labels_1 = ['Survived','Not Survived']

ax1.pie(slices_1, autopct='%1.1f%%',wedgeprops={'edgecolor':'black'}, textprops={'fontsize':14}, labels=labels_1)



slices_2 = [len(df[df['Pclass']==1]), len(df[df['Pclass']==2]), len(df[df['Pclass']==3])]

labels_2 = ['Pclass 1', 'Pclass 2', 'Pclass 3']

ax2.pie(slices_2, labels=labels_2, autopct='%1.1f%%', wedgeprops={'edgecolor':'black'}, textprops={'fontsize':14})



plt.show()
sns.catplot(y='Age',x='Survived',data=df,kind='violin')

plt.show()
fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



ax1.hist(df['Age'],edgecolor='black',bins=30)

ax1.set_title('AGE')



ax2.hist(df['Fare'],edgecolor='black',bins=30)

ax2.set_title('FARE')



plt.show()
df.isnull().sum()
print('avg age of pclass 1: {}'.format(df.Age.loc[df.Pclass==1].mean()))

print('avg age of pclass 2: {}'.format(df.Age.loc[df.Pclass==2].mean()))

print('avg age of pclass 3: {}'.format(df.Age.loc[df.Pclass==3].mean()))

sns.boxplot(x='Pclass', y='Age', data=df)
def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return 38.23

        elif pclass == 2:

            return 29.87

        elif pclass == 3:

            return 25.14

    else:

        return age
def preprocess(df):

    df.Age = df[['Age','Pclass']].apply(impute_age,axis=1)

    sex = pd.get_dummies(df['Sex'], drop_first=True)

    embarked = pd.get_dummies(df['Embarked'], drop_first=True)

    df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    df = pd.concat([df,sex,embarked],axis=1)

    return df
df = preprocess(df)

df.head()
X = df.drop('Survived',axis=1)

y = df.Survived
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,y)
df_test = test_data.copy()

df_test = preprocess(df_test)

df_test.head()
predictions = model.predict(df_test.fillna(method='pad'))
output = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})

output.to_csv('My_Submission',index=False)

print('Your Submission is finally saved.')