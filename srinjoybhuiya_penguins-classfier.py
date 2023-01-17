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
penguins_iter=pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_lter.csv')

penguins_size=pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')
penguins_size.head()
df=penguins_size.copy()
import matplotlib.pyplot as plt 

import seaborn as sns



df=df.dropna()

sns.heatmap(df.isna())

plt.show()
sns.heatmap(df.corr())
# One Hot encoding of the  columns

target = 'species'

encode = ['sex', 'island']



for col in encode:

    dummy = pd.get_dummies(df[col], prefix=col)

    df = pd.concat([df, dummy], axis=1)

    del df[col]



# value encoding of the target

target_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}





def target_encode(val):

    return target_map[val]





df['species'] = df['species'].apply(target_encode)
X=df.drop(columns='species')

Y=df.species
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, Y, train_size=0.8,test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()

clf.fit(X_train,y_train)
import sklearn

prediction=clf.predict(X_test)

sklearn.metrics.confusion_matrix(y_test, prediction)
sklearn.metrics.accuracy_score(y_test, prediction)
prediction_train=clf.predict(X_train)

sklearn.metrics.accuracy_score(y_train, prediction_train)

