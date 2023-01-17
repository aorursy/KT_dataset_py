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
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2.5)

import missingno as msno
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_xx = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_xx.astype(int)
df_train.head()
df_train.describe()
df_train.max()
df_test.describe()
df_test.shape

for col in df_test.columns:
    msg= 'column: {:=>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)
msno.matrix(df=df_train.iloc[:, :] , figsize=(8,8), color = ( 1, 0.5 , 0.2))
msno.bar(df=df_train.iloc[:, :] , figsize=(8,8), color = ( 1, 0.5 , 0.2))
f, ax = plt.subplots(1,2, figsize =(18,5))

df_train['Survived'].value_counts().plot.pie(explode=[1,1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')
plt.show(f)
df_train.shape
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap='cool')

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
f, ax = plt.subplots(1, 2  ,figsize=(18 , 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data= df_train, ax = ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex']).mean()
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data= df_train, size=5, aspect=1)
plt.show()
sns.factorplot(x='Sex' , y='Survived', hue='Pclass', data=df_train, saturation = 5,
              size=8, aspect=1)
print('high age : {:.1f} years'.format(df_train['Age'].max()))
print('low age : {:.1f} years'.format(df_train['Age'].min()))
print('avg age : {:.1f} years'.format(df_train['Age'].mean()))
foo = df_train['Age']
fig, ax = plt.subplots(1,1, figsize= (9,5))
sns.kdeplot(df_train[(df_train['Survived'] == 1)&(df_train['Pclass'] == 1)]['Age'], ax=ax)
sns.kdeplot(df_train[(df_train['Survived'] == 0)&(df_train['Pclass'] == 1)]['Age'], ax =ax )
plt.legend(['Survived = 1', 'Survived = 0'])
plt.show()

#     f = plt.figure(figsize=(10,10))
#     f, ax = plt.subplots(1,1, figsize=(10,10))
    
#     plt.figure(figsize=(10,10))
df_train[(df_train['Survived'] == 1)&(df_train['Pclass'] == 1)]['Age']

# f, ax = plt.subplots(1,1, figsize=(10,10))
# a = np.arange(100)
# b = np.sin(a)
# ax.plot(b)
# plt.show()
# plt.figure(figsize=(5,5))
# a = np.arange(100)
# b = np.sin(a)

# plt.plot(b)
plt.figure(figsize=(5, 5))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('age')
plt.legend(['1st', '2nd', '3rd'])
change_age_range_survival_ratio = []

for i in range(1, 81):
    change_age_range_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age']< i ]['Survived']))

plt.figure(figsize=(7,7))
plt.plot(change_age_range_survival_ratio)
plt.title('Survival range of Age', y= 1.1)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()
a = []

for i in range(1, 81):
    a.append(df_train)
round(df_train[df_train['Age'] < 10 ]['Survived'].sum()/len(df_train[df_train['Age'] < 10]['Survived']),1)
f, ax = plt.subplots(1, 1, figsize=(9, 4))
sns.violinplot('Pclass', 'Age', hue= 'Survived', data= df_train, scale='area', split= True,)
ax.set_title('pclass and Age vs Survived')
ax.set_yticks(range(0, 110, 20))


# sns.violinplot('Sex', 'Age', hue='Survived', data = df_train, scale='count', split= False, ax=ax[1])
# ax[1].set_title('Sex and jAge vs Survived')
# ax[1].set_yticks(range(0,110, 10))
plt. show()
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data= df_train, ax=ax[0,0])
ax[0,0].set_title('Passengers Board')

sns.countplot('Embarked', hue='Sex', data = df_train, ax=ax[0,1])
ax[0,1].set_title('Sex split for embarked')

sns.countplot('Embarked', hue ='Survived', data= df_train, ax=ax[1,0])
ax[1,0].set_title('Survived vs Embarked')

sns.countplot('Embarked', hue = 'Pclass', data = df_train, ax=ax[1,1])
ax[1,1].set_title('Pclass vs Embarked')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] +1
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] +1
print('Maximum size of Family:', df_train["FamilySize"].max())
print('Maximum size of Family:', df_train["FamilySize"].min())
f ,ax = plt.subplots(1,3, figsize= (35, 10))
sns.countplot('FamilySize', data= df_train, ax=ax[0])
ax[0].set_title('(1)No. Of Passenger Boarded', y=1.02)
ax[0].set_ylabel('')

sns.countplot('FamilySize', hue = 'Survived',data = df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamiltSize', y=1.02)
ax[1].set_ylabel('')

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index= True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
df_train
fig, ax = plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df_train['Fare'], color='b', label = 'Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc = 'best')
#왜도 가 왼쪽으로 기울었다
df_train['Fare'].skew()
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)
df_train['Ticket'].value_counts()
df_train['Age'].isnull().sum()
df_train['initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['initial'] =df_test.Name.str.extract('([A-Za-z]+)\.')
df_train
df_train.Name.str.extract('([A-Za-z]*)\.').head(10)
pd.crosstab(df_train['initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
pd.crosstab(df_train['initial'], df_train['Sex']).T
df_train['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
                           ['Miss', 'Miss', 'Miss', 'Mr','Mr','Mrs','Mrs', 'Other','Other', 'Other', 'Mr','Mr','Mr','Mr'], inplace = True)
df_test['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],
                           ['Miss', 'Miss', 'Miss', 'Mr','Mr','Mrs','Mrs', 'Other','Other', 'Other', 'Mr','Mr','Mr','Mr'], inplace = True)
df_train.groupby('initial').mean()
df_train.groupby('initial')['Survived'].mean().plot.bar()
df_all = pd.concat([df_train, df_test])
df_all.reset_index(drop=True)
df_all.groupby('initial').mean()
df_train.loc[:, :]
df_train.loc[(df_train["Age"].isnull()) & (df_train['initial'] == 'Mr') , 'Age' ] = 33
df_train.loc[(df_train["Age"].isnull()) & (df_train['initial'] == 'Mrs') , 'Age' ] = 37
df_train.loc[(df_train["Age"].isnull()) & (df_train['initial'] == 'Master') , 'Age' ] = 5
df_train.loc[(df_train["Age"].isnull()) & (df_train['initial'] == 'Miss') , 'Age' ] = 22
df_train.loc[(df_train["Age"].isnull()) & (df_train['initial'] == 'Other') , 'Age' ] = 45

df_test.loc[(df_test["Age"].isnull()) & (df_test['initial'] == 'Mr') , 'Age' ] = 33
df_test.loc[(df_test["Age"].isnull()) & (df_test['initial'] == 'Mrs') , 'Age' ] = 37
df_test.loc[(df_test["Age"].isnull()) & (df_test['initial'] == 'Master') , 'Age' ] = 5
df_test.loc[(df_test["Age"].isnull()) & (df_test['initial'] == 'Miss') , 'Age' ] = 22
df_test.loc[(df_test["Age"].isnull()) & (df_test['initial'] == 'Other') , 'Age' ] = 45
# df_train.Age.isnull().sum()
df_test.Age.isnull().sum()
df_train.Embarked.isnull().sum()
df_train.Embarked.fillna('S', inplace = True)
df_train.Embarked.isnull().sum()
df_train['Age_cat']= 0
df_train.head()
df_train.loc[df_train['Age'] < 10 , 'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20) , 'Age_cat'] =1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30) , 'Age_cat'] =2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40) , 'Age_cat'] =3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50) , 'Age_cat'] =4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60) , 'Age_cat'] =5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70) , 'Age_cat'] =6
df_train.loc[(70 <= df_train['Age']) , 'Age_cat'] =7
df_test.loc[df_test['Age'] < 10 , 'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20) , 'Age_cat'] =1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30) , 'Age_cat'] =2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40) , 'Age_cat'] =3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50) , 'Age_cat'] =4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60) , 'Age_cat'] =5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70) , 'Age_cat'] =6
df_test.loc[(70 <= df_test['Age']) , 'Age_cat'] =7
df_test.head()
def category_age(x):
    if x < 10:
        return 0
    elif x <20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
df_train['Age_cat2'] = df_train['Age'].apply(category_age)
(df_train['Age_cat'] == df_train['Age_cat2']).all()
type(df_train.Age_cat2)
df_train.head()
df_train.drop(['Age', 'Age_cat2'], axis = 1, inplace = True)
df_test.drop(['Age' ], axis = 1, inplace = True)
###############################
df_train.initial.unique()

df_train['initial'] = df_train['initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['initial'] = df_test['initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_train.Embarked.unique()
df_train.Embarked.value_counts()
df_train['Emabarked'] = df_train['Embarked'].map({'C': 0, 'Q' : 1 , 'S' : 2})
df_test['Emabarked'] = df_test['Embarked'].map({'C': 0, 'Q' : 1 , 'S' : 2})

df_train
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})

heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'initial', 'Age_cat']]
print(heatmap_data)
colormap = plt.cm.BuGn
plt.figure(figsize = (12, 10))
plt.title('Pearson Correalation of Features', y=1.05 , size = 15)
sns.heatmap(heatmap_data.corr(), linewidths = 0.1 , vmax = 1.0,
           square = True, cmap = colormap, linecolor = 'white', annot = True , annot_kws={'size': 16}, fmt='.2f')
df_train = pd.get_dummies(df_train, columns = ['initial'], prefix = 'initial')
df_test = pd.get_dummies(df_test, columns = ['initial'], prefix = 'initial')
df_train = pd.get_dummies(df_train, columns = ['Embarked'], prefix = 'Embarked')
df_test = pd.get_dummies(df_test, columns = ['Embarked'], prefix = 'Embarked')
df_test.head()
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch','Ticket','Cabin','initial_0','initial_1','initial_3','initial_4','Embarked_C', 'Embarked_Q', 'Embarked_S','Age_cat'], axis = 1, inplace = True)
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch','Ticket','Cabin','initial_0','initial_1','initial_3','initial_4','Embarked_C', 'Embarked_Q', 'Embarked_S','Age_cat'], axis = 1, inplace = True)
df_test.Fare = df_test.Fare.fillna(df_test.Fare.mean())
# X_test = X_test.fillna(X_train.mean())
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train = df_train.drop('Survived', axis =1 ).values
target_label = df_train['Survived'].values
X_test = df_test.values
X_tr, X_vid, y_tr, y_vid =  train_test_split(X_train, target_label, test_size = 0.25, random_state = 2018)

model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vid)
from pandas import Series
print(f'총 {y_vid.shape[0]}명 중 {100*metrics.accuracy_score(prediction, y_vid):.2f}% 정확도로 생존 맞춤')
feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index = df_test.columns)
type(df_test)
df_train.columns
df_test.columns
# df_test = pd.DataFrame(df_test, columns = ['Survived','Pclass', 'Sex', 'Fare', 'Age_cat', 'initial_0', 'initial_1',
#        'initial_2', 'initial_3', 'initial_4', 'Embarked_C', 'Embarked_Q',
#        'Embarked_S'])
plt.figure(figsize = (8,8))
Series_feat_imp.sort_values(ascending = True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission = submission['Survived']
prediction2 = model.predict(X_test)
prediction2.shape
print(f'총 {submission.shape[0]}명 중 {100*metrics.accuracy_score(prediction2, submission):.2f}% 정확도로 생존 맞춤')
