import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

display(train_set.head())

display(train_set.isnull().sum())

display(train_set.info())

display(train_set.describe())

display(train_set.describe(include=['O']))
plt.figure(figsize=(12,6))



plt.subplot(1,2,1)

sns.boxplot(x='Sex',y='Age',data=train_set)



plt.subplot(1,2,2)

sns.boxplot(x='Sex',y='Age',hue='Survived',data=train_set)
def fill_ages(data):

    age = data[0]

    sex = data[1]

    if np.isnan(age):

        if sex is 'male':

            return 29

        else:

            return 25

    else:

        return age



def pre_process_ML_data(df):

    df = df.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)

    df['Age'] = df[['Age','Sex']].apply(fill_ages,axis=1)

    #df['Embarked'].fillna('S',inplace = True)

    emb_dum = pd.get_dummies(df['Embarked'],drop_first=True)

    sex_dum = pd.get_dummies(df['Sex'],drop_first=True)

    df = df.drop(['Embarked','Sex'],axis=1)

    df = pd.concat([df,emb_dum,sex_dum],axis=1)

    return df



plt.figure(figsize=(12,8))

X = pre_process_ML_data(train_set)

#display(X.info())

#display(X.describe())

display(sns.heatmap(X.corr(),cmap='coolwarm'))

X = X.drop('Survived',axis=1).values

y = train_set['Survived'].values

print(X)

print(y)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = LogisticRegression().fit(X_train,y_train)
plt.figure(figsize=(10,6))

prediction = classifier.predict(X_test)

conf_mat = confusion_matrix(y_test,prediction)

print(conf_mat)

display(sns.heatmap(data=conf_mat,cmap='coolwarm',annot=True))

print(classification_report(y_test,prediction))
test_set = pd.read_csv('/kaggle/input/titanic/test.csv')

p_ids = test_set['PassengerId']
p_ids.nunique()
test_set.describe()

test_set.isnull().sum()
X = pre_process_ML_data(test_set).values



pred = classifier.predict(X_test)
pd.DataFrame(pred).info()
submit = pd.DataFrame({'PassengerId':p_ids,'Survived':pred})

submit.head()

submit.to_csv('gender_submission.csv',ignore_index=True)