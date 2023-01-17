import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from keras.utils import np_utils

%matplotlib inline

import check_miss_value

# csv ファイルから読み込み
train_csv = pd.read_csv('./train.csv')
test_csv = pd.read_csv('./test.csv')
# train Survived あり
# test PassengerId あり
train_csv.shape, test_csv.shape
train_csv.info()
test_csv.info()
train_csv.describe()
train_csv.corr()
train_csv.corr()[train_csv.corr() < -0.2]
train_csv.corr()[train_csv.corr() > 0.2]
sns.countplot(x='Survived', data=train_csv)
# 生存者の割合
train_csv.Survived.sum() / train_csv.Survived.count()
train_csv.Survived[train_csv.Survived == 1].count() / train_csv.Survived.count()
train_csv.groupby(['Survived', 'Sex'])['Survived'].count()
sns.catplot(x='Sex', col='Survived', data=train_csv, kind='count')
train_csv.columns
# 円グラフ
f, ax = plt.subplots(1, 2, figsize=(16, 8))
f.patch.set_facecolor('white')
train_csv['Survived'][train_csv['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
train_csv['Survived'][train_csv['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.2], autopct='%1.1f%%', ax=ax[1], shadow=True)
ax[0].set_title('Survived(male)')
ax[1].set_title('Survived(female)')
train_csv['Survived'][train_csv['Sex'] == 'female'].value_counts()
# cross table
pd.crosstab(train_csv.Pclass, train_csv.Survived, margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
pd.crosstab(train_csv.Pclass, train_csv.Survived, margins=True, margins_name='Total')
sns.catplot('Pclass', 'Survived', kind='point', data=train_csv)
pd.crosstab([train_csv.Survived], train_csv.Pclass, margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
pd.crosstab([train_csv.Sex, train_csv.Survived], train_csv.Pclass, margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
sns.catplot('Pclass', 'Survived', hue='Sex', kind='point', data=train_csv)
sns.catplot(x='Survived', col='Embarked', kind='count', data=train_csv)
sns.catplot('Embarked', 'Survived', kind='point', data=train_csv)
sns.catplot('Embarked', 'Survived', hue='Sex', kind='point', data=train_csv)
sns.catplot('Embarked', 'Survived', col='Pclass', hue='Sex', kind='point', data=train_csv)
pd.crosstab([train_csv.Survived], [train_csv.Sex, train_csv.Pclass, train_csv.Embarked], margins=True)
train_csv.Age.min(), train_csv.Age.max()
sns.distplot(train_csv.Age, bins=9, kde=True)
sns.pairplot(train_csv)
train_csv['Age_bin'] = train_csv.Age // 10
sns.catplot('Age_bin', 'Survived', hue='Sex', kind='point', data=train_csv)
sns.catplot('Age_bin', 'Survived', col='Pclass', row='Sex', kind='point', data=train_csv)
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.Age_bin, train_csv.Pclass], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
sns.catplot('SibSp', 'Survived', col='Pclass', row='Sex', kind='point', data=train_csv)
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.SibSp, train_csv.Pclass], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
sns.catplot('Parch', 'Survived', col='Pclass', row='Sex', kind='point', data=train_csv)
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.Parch, train_csv.Pclass], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.Pclass, train_csv.Parch], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
train_csv.Fare.min(), train_csv.Fare.max()
sns.distplot(train_csv.Fare)
train_csv['Fare_bin'] = train_csv.Fare // 20
train_csv.Fare_bin.value_counts()
sns.catplot('Fare_bin', 'Survived', col='Pclass', row='Sex', kind='point', data=train_csv)
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.Fare_bin, train_csv.Pclass], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')
pd.crosstab([train_csv.Sex, train_csv.Survived], [train_csv.Pclass, train_csv.Fare_bin], margins=True).style.background_gradient(cmap='autumn_r').set_properties(color='black')

