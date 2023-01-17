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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', encoding='utf-8')

df.head()
our_set = df[df['neighbourhood_group'].isin(['Brooklyn', 'Manhattan'])]

our_set.shape, df.shape
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15,8))

sns.distplot(our_set['price'])

plt.title("Price per night")

plt.xlabel('Dollar Amount')

plt.ylabel('Ratio')
set(our_set['room_type'])
import random



random_seed = random.randint(0,1000)
our_set = our_set.replace('Brooklyn', 0).replace('Manhattan', 1)
x = pd.get_dummies(our_set['room_type'])

core_cols = our_set[['neighbourhood_group', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']]

combined = core_cols.merge(x, left_index=True, right_index=True, how='left').reset_index()

combined.head()
final = combined.drop(columns=['index']).dropna()

final.shape
import missingno as msno

msno.matrix(final)
final.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(final[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']], final['neighbourhood_group'], 

                                                    test_size=0.25, random_state=random_seed)
from sklearn.ensemble import RandomForestClassifier

random_seed = random.randint(0,1000)

rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=random_seed)

rf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



prediction = rf.predict(X_test)

accuracy = accuracy_score(y_test, prediction)



print(f'Mean accuracy score: {accuracy:.3}')
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_test, prediction), columns=['Brooklyn', 'Manhattan'], index=['Brooklyn', 'Manhattan'])

sns.heatmap(cm, annot=True, fmt='d')