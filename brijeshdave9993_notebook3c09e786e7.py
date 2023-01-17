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
import sklearn

import numpy as np

import pandas as pd

import seaborn as sn

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df.head()
df.info()
df['quality'].value_counts()
corr = df.corr()

plt.figure(figsize=(16,16))

sn.heatmap(corr,annot=True)

plt.figure(figsize=(10,10))

plt.show()
from sklearn.neighbors import KNeighborsClassifier as KNN



df1 = df.drop(columns=['quality'])

df2 = df[['quality']]

X = df1.values

y = df2.values



from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30)

print('Training size: {}, Testing side: {}'.format(X_train.size,X_test.size))
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import classification_report

clf = KNN(1)

clf.fit(X_train, y_train)



KNN_predictions = clf.predict(X_test)

print("Predictions:",KNN_predictions[:10])

print("Actual:",y_test[:10])



print(classification_report(y_test, KNN_predictions, digits=3))