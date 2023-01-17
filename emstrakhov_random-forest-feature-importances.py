# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cardio_train.csv', sep=';', index_col=0)

df.head()
df['age'] = np.floor(df['age'] / 365.25)

# df['gender'] = df['gender'].map({1:0, 2:1})

# new_df = pd.get_dummies(df, columns=['cholesterol', 'gluc'])

# new_df.head()

df.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(df.drop('cardio', axis=1),

                                                      df['cardio'],

                                                      test_size=0.3,

                                                      random_state=2019)
# Обучение случайного леса

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state=2019)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_valid)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_valid, y_pred))
# Доводка параметров
import matplotlib.pyplot as plt



features = dict(zip(range(len(df.columns)-1), df.columns[:-1]))



# Важность признаков

importances = rf.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(df.columns[:-1]))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);