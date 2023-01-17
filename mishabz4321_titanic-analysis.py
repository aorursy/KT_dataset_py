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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train= pd.read_csv('/kaggle/input/titanic/train.csv')

test_data= pd.read_csv('/kaggle/input/titanic/test.csv')

test= test_data.copy()
train.head()
train.info()
train.describe().transpose()
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
train.isnull().sum()
train.groupby('Pclass')['Age'].mean()
def impute_age(col):

    Age= col[0]

    Pclass= col[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        elif Pclass== 2:

            return 30

        else:

            return 25

    else:

        return Age
train['Age']= train[['Age','Pclass']].apply(impute_age, axis=1)

test['Age']= test[['Age','Pclass']].apply(impute_age,axis=1)
train= train.drop('Cabin', axis=1)

test= test.drop('Cabin', axis=1)
test.isnull().sum()
test= test.fillna(test.mean())

test.isnull().sum()
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
survival_count= train.groupby('Survived')['PassengerId'].count().to_frame().reset_index().rename(columns={'PassengerId':'count'})
survival_count
print('% of passengers survived:', (survival_count['count'][1]*100)/891)

print('% of passengers not survived:', (survival_count['count'][0]*100)/891)
df1= train.copy()

fig, axes= plt.subplots(ncols=2, nrows=1, figsize=(13, 3))

plt.tight_layout()

df1.groupby('Survived')['PassengerId'].count().plot(kind='pie', labels=['38.38% not survived','61.61% survived'], ax=axes[0])

sns.countplot(df1['Survived'], ax=axes[1])

axes[0].set_ylabel('')

axes[1].set_xticklabels(['38.38% not survived','61.61% survived'])

axes[0].set_title('Target distribution')

axes[1].set_title('Target count')
plt.figure(figsize=(10,8))

sns.boxplot(x='Survived',y='Age',data=df1)
sns.catplot(x='Pclass',data=df1, col='Survived',kind='count', palette='Blues_d')
sns.set_style('whitegrid')

plt.figure(figsize=(10,8))

sns.distplot(df1['Age'], bins=30)
plt.figure(figsize=(10,8))

sns.boxplot(x='Pclass',y='Age',data=df1)
sns.distplot
plt.figure(figsize=(13,6))

sns.distplot(df1[df1['Sex']=='male']['Age'], kde_kws={'label':'male','color':'r'})

sns.distplot(df1[df1['Sex']=='female']['Age'], kde_kws={'label':'female','color':'b'})

plt.legend()

plt.show()
plt.figure(figsize=(10,8))

sns.violinplot(x='Sex',y='Age',data=df1,hue='Pclass',palette='muted')
sns.set_palette('Paired')

sns.catplot(x='Pclass',data=df1,kind='count',col='Sex', hue='Survived')
df1.groupby('SibSp')['PassengerId'].count()
df1.groupby('Parch')['PassengerId'].count()
sns.set_palette('Set1')

sns.countplot(df1['SibSp'])
sns.set_palette('Paired')

sns.catplot(x='SibSp',data=df1,kind='count',col='Sex', hue='Survived')
sns.set_palette('Set1')

sns.countplot(df1['Parch'])
sns.set_palette('Paired')

sns.catplot(x='Parch',data=df1,kind='count',col='Sex', hue='Survived')
plt.figure(figsize=(10,8))

sns.set_palette('dark')

sns.scatterplot(x='Fare', y='Age', data=df1)
plt.figure(figsize=(20,6))

sns.distplot(df1[df1['Pclass']==1]['Fare'], bins=30, kde_kws={'label':'1st class', 'color':'r'})

sns.distplot(df1[df1['Pclass']==2]['Fare'], bins=30, kde_kws={'label':'2nd class', 'color':'b'})

sns.distplot(df1[df1['Pclass']==3]['Fare'], bins=30, kde_kws={'label':'3rd class', 'color':'g'})
sns.countplot(df1['Embarked'])
sns.catplot(kind='count', x='Embarked', data=df1, col='Sex')
sns.catplot(kind='count', x='Embarked', data=df1, col='Sex', hue='Survived', palette='Set1')
plt.figure(figsize=(15,8))

sns.swarmplot(x='Embarked', y='Age', data=df1, palette='Set2', hue='Sex', dodge=True)
plt.figure(figsize=(15,8))

sns.swarmplot(x='Embarked', y='Fare', data=df1, palette='Set2', hue='Pclass', dodge=True)
df1['Ticket'].value_counts()
df1['Ticket'].value_counts().head(20)
df2= df1.copy()
df2.groupby('Pclass')['Age'].mean()
sns.boxplot(x='Pclass', y='Age', data=df2)
df2_class1= df2[df2['Pclass']==1]

df2_class2= df2[df2['Pclass']==2]

df2_class3= df2[df2['Pclass']==3]
plt.figure(figsize=(16,8))

sns.distplot(df2_class1['Age'],kde_kws={'label':'1st class'})

sns.distplot(df2_class2['Age'],kde_kws={'label':'2st class'})

sns.distplot(df2_class3['Age'],kde_kws={'label':'3st class'})
from scipy import stats

def outlier_remove_age(df):

    df_out= pd.DataFrame()

    for key, subdf in df2.groupby('Pclass'):

        subdf['z_score_price']= np.abs(stats.zscore(subdf['Age']))

        reduced_df= subdf[(subdf.z_score_price>-2)&(subdf.z_score_price<2)]

        df_out= pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out
df3= outlier_remove_age(df2)
df2.shape
df3.shape
sns.boxplot(x='Pclass', y='Age', data=df3)
df3.head()
df3[df3['Fare']==0]
df4= df3.copy()

df4= df3[df3['Fare'] != 0]
df3.shape
df4.shape
df4[df4['Age']<10]
df4= df4[df4['PassengerId']!=778]
df4.shape
df4.groupby('Pclass')['Fare'].mean()
sns.boxplot(x='Pclass', y='Fare', data=df4)
from scipy import stats

def outlier_remove_fare(df):

    df_out= pd.DataFrame()

    for key, subdf in df4.groupby('Pclass'):

        subdf['z_score_fare']= np.abs(stats.zscore(subdf['Fare']))

        reduced_df= subdf[(subdf.z_score_fare>-3)&(subdf.z_score_fare<3)]

        df_out= pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out
df5= outlier_remove_fare(df4)
df5.shape
sns.boxplot(x='Pclass', y='Fare', data=df5)
df5.head()
df5.groupby('Sex')['Age'].mean()
plt.figure(figsize=(20,14))

sns.set_palette('dark')

sns.scatterplot(x='Fare', y='Age', data=df5, hue='Sex')
plt.figure(figsize=(10,8))

sns.set_palette('dark')

sns.color_palette('Set1')

sns.swarmplot(x='Embarked', y='Fare', data=df5)
def outlier_remove_fare(df):

    df_out= pd.DataFrame()

    for key, subdf in df5.groupby('Embarked'):

        subdf['z_score_fare_embarked']= np.abs(stats.zscore(subdf['Fare']))

        reduced_df= subdf[(subdf.z_score_fare_embarked>-3)&(subdf.z_score_fare_embarked<3)]

        df_out= pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out
df6= outlier_remove_fare(df5)
df5.shape
df6.shape
plt.figure(figsize=(10,8))

sns.set_palette('dark')

sns.color_palette('Set1')

sns.swarmplot(x='Embarked', y='Fare', data=df6)
df6.head()
df6['Name']= df6['Name'].apply(lambda x: x.strip())

df6['Position']= df6['Name'].apply(lambda x: x.split()[1])

test['Name']= test['Name'].apply(lambda x: x.strip())

test['Position']= test['Name'].apply(lambda x: x.split()[1])
position_stats= df6['Position'].value_counts()

position_stats_lessthan_10= position_stats[position_stats<10]

position_stats_test= test['Position'].value_counts()

position_stats_lessthan_10_test= position_stats_test[position_stats_test<10]
df6['Position']= df6['Position'].apply(lambda x: 'other' if x in position_stats_lessthan_10 else x)

test['Position']= test['Position'].apply(lambda x: 'other' if x in position_stats_lessthan_10_test else x)
df6.Position.value_counts()
sns.countplot(df6['Position'], order=['Mr.', 'Miss.', 'Mrs.', 'other', 'Master.'], palette='Blues_d')
sns.countplot(df6['Position'], order=['Mr.', 'Miss.', 'Mrs.', 'other', 'Master.'], palette='Blues_d', hue=df6['Survived'])
df6.groupby(['Position','Survived'])['PassengerId'].count().to_frame().rename(columns={'PassengerId':'count'})
df7= df6.drop(['PassengerId','Name','Ticket','z_score_price','z_score_fare','z_score_fare_embarked'], axis=1)

test= test.drop(['PassengerId','Name','Ticket'], axis=1)

df7.head()
sex= pd.get_dummies(df7['Sex'],drop_first=True)

sex_test= pd.get_dummies(test['Sex'],drop_first=True)

embarked= pd.get_dummies(df7['Embarked'], drop_first=True)

embarked_test= pd.get_dummies(test['Embarked'], drop_first=True)

position= pd.get_dummies(df7['Position'])

position= position.drop('other', axis=1)

position_test= pd.get_dummies(test['Position'])

position_test= position_test.drop('other', axis=1)
df8= df7.drop(['Sex','Embarked','Position'], axis=1)

test= test.drop(['Sex','Embarked','Position'], axis=1)

df8.head()
test.head()
df9= pd.concat([df8, sex,embarked, position], axis=1)

test= pd.concat([test, sex_test,embarked_test, position_test], axis=1)

df9.head()
test.head()
X= df9.drop('Survived', axis=1)

y=df9.Survived
X.head()
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaled_X= scaler.fit_transform(X)

scaled_test= scaler.transform(test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(scaled_X,y,test_size=0.2, random_state=101)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
lr.fit(X_train,y_train)

pred= lr.predict(X_test)

score= lr.score(X_test,y_test)
score
from sklearn.model_selection import cross_val_score, ShuffleSplit

cv= ShuffleSplit(n_splits=5,random_state=0, test_size=0.2)

score_lr=np.mean(cross_val_score(LogisticRegression(), scaled_X, y, cv=cv))
score_lr
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
naive_model= GaussianNB()

naive_model.fit(X_train,y_train)

y_pred= naive_model.predict(X_test)
cv= ShuffleSplit(n_splits=5,random_state=0, test_size=0.2)

score_naive= np.mean(cross_val_score(GaussianNB(), scaled_X, y, cv=cv))

score_naive
from sklearn.ensemble import RandomForestClassifier

model_forest= RandomForestClassifier(n_estimators=200, max_depth=10, random_state=101)
cv= ShuffleSplit(n_splits=5,random_state=0, test_size=0.2)

score_forest= np.mean(cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=101), scaled_X, y, cv=cv))

score_forest
lr.fit(scaled_X, y)

predictions= lr.predict(scaled_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")