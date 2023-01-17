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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

# Presentation of the datas

data.info()



# Repartition 

plt.figure(figsize=[5,5])

sns.set(style='darkgrid')

ax = sns.countplot(x='blueWins', data=data, palette='Set3')

data.loc[:,'blueWins'].value_counts()

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression









data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

#data.info()



data.drop(['gameId', 'blueFirstBlood', 'blueEliteMonsters', 'blueDragons',

           'redFirstBlood', 'redEliteMonsters', 'redDragons'], 

            axis=1, inplace=True)





y = data.blueWins.values

x_data = data.drop(['blueWins'], axis=1)



x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=5)



# Make a first prediction

# Defining the model

lr = LogisticRegression()

# Training the model:

lr.fit(x_train, y_train)

# Predicting target values by using x_test and our model:

y_pred0 = lr.predict(x_test)

score_lr = lr.score(x_test, y_test)

print('score Logistic Regression :', score_lr)