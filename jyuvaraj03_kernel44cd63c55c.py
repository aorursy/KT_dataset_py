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

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df = df.drop(['Serial No.'], axis=1)

df.head()
import matplotlib.pyplot as plt
x = df['CGPA']

y = df['Chance of Admit ']

# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)

plt.scatter(x,y)
mask = np.random.rand(len(df)) < 0.7

train = df[mask]

test = df[~mask]

# print(test.head())
from sklearn import linear_model

regr = linear_model.LinearRegression()

X_train = train.drop(['Chance of Admit ', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'Research'], axis=1)

y_train = train[['Chance of Admit ']]

regr.fit (X_train, y_train)

plt.scatter(X_train,y_train)

plt.plot(X_train,regr.coef_*X_train+regr.intercept_,'-r')
X_test = test.drop(['Chance of Admit ', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'Research'], axis=1)

y_test = test[['Chance of Admit ']]

pred = regr.predict(X_test)

print((np.sqrt(mean_squared_error(y_test, pred))))
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# X = df.drop(['Chance of Admit '], axis=1)

X = df.drop(['Chance of Admit '], axis=1)

y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)

print(X_train.head())

# print(y_train.head())
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print((np.sqrt(mean_squared_error(y_test, predictions))))
classifier = RandomForestRegressor()

classifier.fit(X,y)

feature_names = X.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = X.columns

importance_frame['Importance'] = classifier.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=True)
plt.barh([1,2,3,4,5,6,7], importance_frame['Importance'], align='center', alpha=0.5)

plt.yticks([1,2,3,4,5,6,7], importance_frame['Features'])

plt.xlabel('Importance')

plt.title('Feature Importances')

plt.show()