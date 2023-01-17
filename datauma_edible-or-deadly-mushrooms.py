# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mushroom = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

mushroom.head()
mushroom.describe()
#Information of mushroom dataset

mushroom.info()

#No missing data
print('The classes of the mushroom dataset are ', np.unique(mushroom['class']))
print('The shape of the mushroom dataset is', mushroom.shape)
def mushroom_graph(name, ax):

    mushroom[name].value_counts().plot(kind='bar', ax=ax, 

                                              color='coral')

    ax.set_alpha(0.8)

    ax.set_title(name.replace('-',' '),fontsize=15)



    # create a list to collect the plt.patches data

    totals = []



    for i in ax.patches:

        totals.append(i.get_height())

    

    total = sum(totals)



    for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

        ax.text(i.get_x()-.03, i.get_height()+20, \

                str(round((i.get_height()/total)*100, 2))+'%', fontsize=10,

                    color='dimgrey')

    



fig1, ax1 = plt.subplots(4,2,figsize=(17,16))

mushroom_graph('cap-color', ax1[0,0])

mushroom_graph('cap-shape', ax1[0,1])

mushroom_graph('stalk-color-below-ring', ax1[1,0])

mushroom_graph('odor', ax1[1,1])

mushroom_graph('stalk-surface-below-ring', ax1[2,0])

mushroom_graph('gill-size', ax1[2,1])

mushroom_graph('population', ax1[3,0])

mushroom_graph('veil-color', ax1[3,1])
#create X and y variable



y = mushroom['class']

X = mushroom.drop('class', axis=1)
#Using get_dummies method, transform the X varible

X = pd.get_dummies(X)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

y = le.fit_transform(y)
#Split the dataset into train and test set



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1, stratify = y)

X_train.shape, X_test.shape, y_train.shape, y_test.shape 
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state=1)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score



print('Accuracy score is %.2f' % accuracy_score(y_test, y_pred))