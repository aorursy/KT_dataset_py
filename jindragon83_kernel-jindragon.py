import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') # 워닝 메세지를 생략해 줍니다. 차후 버전관리를 위해 필요한 정보라고 생각하시면 주석처리 하시면 됩니다.



%matplotlib inline
os.listdir("../input")
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.shape, df_test.shape
df_train.columns
df_train.head()
df_test.head()
df_train.dtypes
df_train.describe()
df_test.describe()
df_train.isnull().sum() / df_train.shape[0]
df_test.isnull().sum() / df_test.shape[0]
f, ax = plt.subplots(1, 2, figsize=(18, 8))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
# pclass 그룹 별 데이터 카운트

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# pclass 그룹 별 생존자 수 합

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# 위와 같은 작업을 crosstab으로 편하게 할 수 있습니다.

pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
# mean은 생존률을 구하게 할 수 있습니다.

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# Age distribution withing classes

plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])