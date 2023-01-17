# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print('The size of the tranin data: ' + str(train.shape))

print('The size of the test data: ' + str(test.shape))
data = pd.concat([train, test], sort = False)
print('The ratio of the survivavility: ' + str(train['Survived'].mean()))
print('Is there null value index into train data?: ' + str(train['Survived'].isnull().sum())) 

print('And, is there any unique value other than 0 or 1?: ' + str(train['Survived'].unique()))
import folium
# Mean latitude/longitude between 3 ports is picked by center.

sea_route = folium.Map(location=[51, -5.76], zoom_start=6)



# I have a little knowledge around there, and got choice them from some good references.

folium.Marker([50.892239,-1.3981697], popup='Southampton<br><i>10 April 1912</i>').add_to(sea_route)

folium.Marker([49.646042,-1.618031], popup='Cherbuorg<br><i>10 April 1912</i>').add_to(sea_route)

folium.Marker([51.853955,-8.2997997], popup='Queenstown<br><i>11 April 1912</i>').add_to(sea_route)



sea_route
print("Embarked is only three port or not?: " + str(train['Embarked'].unique()))

print("And, is there null value?: " + str(train['Embarked'].isnull().sum()))
train['Embarked'].value_counts()
# Null value cells will be changed by the major attribute.

train['Embarked'].fillna('S', inplace=True)
train['Survived'].groupby(train['Embarked']).mean()
# Counting for each embarking ports

sns.countplot(train['Embarked'], hue=train['Survived'])
train['Pclass'].isnull().sum()
# Let me focus on passenger's social level (?) based on their ticket rank.

sns.countplot(train['Embarked'], hue=train['Pclass'])
# Before heatmap making, drop impossible features.

train_heat = train.copy()

train_heat.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age'], 1, inplace=True)

train_heat.head()
train_heat['Sex'].unique()
train_heat['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

train_heat['Embarked'].replace( ['S', 'C', 'Q'], [0, 1, 2], inplace=True )

train_heat.head()
plt.figure(figsize=(14, 12))

sns.heatmap(

    train_heat.astype(float).corr()

    , linewidths=0.1

    , square=True

    , linecolor='white'

    , annot=True

    , cmap='YlGn'

)

plt.show()