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
data = pd.read_csv('/kaggle/input/the-ultimate-halloween-candy-power-ranking/candy-data.csv' )
data.info()
data.head()
from sklearn.model_selection import train_test_split

df = data

X = df.drop('chocolate',axis=1).drop('competitorname', axis=1).values

y = df['chocolate'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('Splitting done. \n')

len(y)
print('Initializing classifier...')

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train) # We fit the Logistic Regression Classifier

predictions = clf.predict(X_test) # We compute the predictions

#print(X_test)

#print(predictions)

us=0

for i in range(len(predictions)):

    print( "prediction: ", predictions[i]," real value: ",y[i])

    if predictions[i]!=y[i]:

        us+=1

print("in ",len(predictions)," prediction ",us," wrong predictions")

    
print("logistic test accurancy:\n",clf.score(X_test,y_test))

 
import matplotlib.pyplot as plt



# Grafik şeklinde ekrana basmak için

plt.plot(X_train, y_train, color='red')

plt.plot(X_test, y_test, color='blue')

plt.show()
