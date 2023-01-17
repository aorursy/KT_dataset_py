import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.head()
print(df.shape)
y_train = df['Survived']

y_train.shape
df.drop(['Survived'],axis=1,inplace=True)

df.info()
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels=False, cbar=True)
df[['Pclass','Age']].groupby(['Pclass'],as_index=False).mean().sort_values(['Age'],ascending=False)
def replace_nan_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if(pd.isnull(Age)):

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 30

        else:

            return 25

    else: 

        return Age
df['Age'] = df[['Age','Pclass']].apply(replace_nan_age, axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=True)
df.drop(columns='Cabin',axis=1,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=True)
df.drop(columns=['Name','Ticket'],axis=1,inplace=True)
Sex = pd.get_dummies(df['Sex'],drop_first=True)

df = pd.concat([df,Sex],axis=1)

df.drop(['Sex'],axis=1,inplace=True)
df['Embarked'].value_counts()
Embarked=pd.get_dummies(df['Embarked'])

df=pd.concat([df,Embarked],axis=1)

df.drop(['Embarked'],axis=1,inplace=True)
df.info()
df.head()
df_test.head()
df_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
Sex = pd.get_dummies(df_test['Sex'],drop_first=True)

df_test = pd.concat([df_test,Sex],axis=1)

df_test.drop(['Sex'],axis=1,inplace=True)
Embarked=pd.get_dummies(df_test['Embarked'])

df_test=pd.concat([df_test,Embarked],axis=1)

df_test.drop(['Embarked'],axis=1,inplace=True)
df_test.info()
df_test.corr()
df_test[['Pclass','Age']].groupby(['Pclass'],as_index=False).mean().sort_values(['Age'],ascending=False)
df[['male','Age']].groupby(['male'],as_index=False).mean().sort_values(['Age'],ascending=False)
def replace_nan_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if(pd.isnull(Age)):

        if Pclass == 1:

            return 41

        elif Pclass == 2:

            return 29

        else:

            return 24

    else: 

        return Age
df_test['Age'] = df_test[['Age','Pclass']].apply(replace_nan_age, axis=1)
df_test.replace(np.nan,df_test['Fare'].mean(),inplace=True)
df_test.info()
print(df.shape)

print(df_test.shape)
x_trn,x_valid,y_trn,y_valid=train_test_split(df,y_train,test_size=0.33,random_state=150)
model=RandomForestClassifier(n_estimators=200,random_state=200,max_features=0.5,min_samples_leaf=3,oob_score=True,n_jobs=-1)

model.fit(df,y_train)
model.score(x_valid,y_valid)
predict_y = model.predict(df_test)
model.score(x_trn,y_trn)
model.feature_importances_

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(x_trn.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure(figsize=(18,12))

plt.title("Feature importances")

plt.bar(range(x_trn.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(x_trn.shape[1]), indices)

plt.xlim([-1, x_trn.shape[1]])

plt.show()
my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predict_y })

my_submission.to_csv('submission.csv', index=False)