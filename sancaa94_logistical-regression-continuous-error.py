

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/data.csv')
df.shape

df['tl_rank'].fillna(df['tl_rank'].mean())
df.head(45)
df.info()
df.columns
df['ride_type_all'].value_counts()
df.describe()
sns.set_style('whitegrid')
sns.countplot(x=df['ride_type_thrill'].replace('Ues', 'Yes'), data=df)
plt.figure(figsize=(10,7))

sns.boxplot(x=df['ride_type_thrill'].replace('Ues', 'Yes'), y=df['tl_rank'])
thrill = pd.get_dummies(df['ride_type_thrill'].replace('Ues', 'Yes'),drop_first=True)

spinning = pd.get_dummies(df['ride_type_spinning'],drop_first=True)

slow = pd.get_dummies(df['ride_type_slow'],drop_first=True)

small_drops = pd.get_dummies(df['ride_type_small_drops'],drop_first=True)

big_drops = pd.get_dummies(df['ride_type_big_drops'],drop_first=True)

dark = pd.get_dummies(df['ride_type_dark'],drop_first=True)

scary = pd.get_dummies(df['ride_type_scary'],drop_first=True)

water = pd.get_dummies(df['ride_type_water'],drop_first=True)

fast_pass = pd.get_dummies(df['fast_pass'],drop_first=True)

classic = pd.get_dummies(df['classic'],drop_first=True)

preschoolers = pd.get_dummies(df['age_interest_preschoolers'],drop_first=True)

kids = pd.get_dummies(df['age_interest_kids'],drop_first=True)

tweens = pd.get_dummies(df['age_interest_tweens'],drop_first=True)

teens = pd.get_dummies(df['age_interest_teens'],drop_first=True)

adults = pd.get_dummies(df['age_interest_adults'],drop_first=True)
adults = pd.get_dummies(df['age_interest_adults'],drop_first=True)

df.drop(['ride_name', 'park_location', 'park_area', 'ride_type_all','ride_type_thrill', 'ride_type_spinning', 'ride_type_slow',

 'ride_type_small_drops', 'ride_type_big_drops', 'ride_type_dark',

       'ride_type_scary', 'ride_type_water', 'fast_pass', 'classic',

       'age_interest_all', 'age_interest_preschoolers', 'age_interest_kids',

       'age_interest_tweens', 'age_interest_teens', 'age_interest_adults','open_date','age_of_ride_total'], axis=1, inplace=True)
Final = pd.concat([df,spinning,slow,small_drops,big_drops,dark,scary,water,fast_pass,classic,preschoolers,kids,tweens,teens,adults],axis=1)
Final
X = Final.drop('tl_rank', axis=1)

y = Final['tl_rank']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)