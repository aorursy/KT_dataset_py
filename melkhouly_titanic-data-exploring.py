import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.preprocessing import LabelBinarizer

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')

train_data.shape
train_data.head()
train_data.isnull().sum()
sb.countplot('Survived',data=train_data)
plt.scatter(train_data['PassengerId'],train_data['Age'],alpha=0.5)
train_data.boxplot(column='Age',by=['Survived', 'Sex'])
one_hot = LabelBinarizer()

one_hot.fit_transform(train_data['Sex'])


o_t = pd.get_dummies(train_data['Sex'])

train_data=train_data.drop('Sex',axis = 1)

train_data=train_data.join(o_t)

train_data.head(100)
df = pd.concat([train_data, test_data], axis=0, sort=True)

# create new Title column

df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

df.head()
df['Title'].value_counts()
# replace rare titles with more common ones

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',

           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',

           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

df.replace({'Title': mapping}, inplace=True)

# confirm that we are left with just six values

df['Title'].value_counts()
# impute missing Age values using median of Title groups

title_ages = dict(df.groupby('Title')['Age'].median())



# create a column of the average ages

df['age_med'] = df['Title'].apply(lambda x: title_ages[x])



# replace all missing ages with the value in this column

df['Age'].fillna(df['age_med'], inplace=True, )

del df['age_med']



# then visualize it

sns.barplot(x='Title', y='Age', data=df, estimator=np.median, ci=None, palette='Blues_d')

plt.xticks(rotation=45)

plt.show()
sns.countplot(x='Title', data=df, palette='hls', hue='Survived')

plt.xticks(rotation=45)

plt.show()