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
df = pd.read_csv('/kaggle/input/coronavirus-worldometers/covid19.csv')

df
import matplotlib.pyplot as plt



plt.scatter(df.population, df.total_tests)

plt.plot(df.population, df.total_tests)
df1 = df.drop(columns=['country'])
X = df1.drop(columns=['new_cases'])

y = df1['new_cases']
#Important. The column new_cases is full of string values, so convert them to numerical values.



for i in range(213):    

    if(type(y[i]) == str):

        a = y[i].replace(',','')

        a = int(a)

        y[i] = a
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.scatter(X_test.population, y_test)

plt.plot(X_test.population, y_test)
from sklearn import ensemble



model = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.01, loss='ls')
model.fit(X_train, y_train)
model.predict(X_test)
from sklearn.metrics import accuracy_score



model.score(X_test, y_test)
pred = model.predict(X_test)
plt.scatter(X_test.population, pred)

plt.plot(X_test.population, pred)
a_pred = model.predict(X)



a_pred
#Accuracy



model.score(X, y)
print("Predictions: \n")



for i in range(212):

    

    print("New cases for",df['country'][i],": ",a_pred[i])

result = pd.DataFrame({"country":df['country'],"new_cases":a_pred})

result
result.to_csv("New_Cases_Predictions.csv", index=False)