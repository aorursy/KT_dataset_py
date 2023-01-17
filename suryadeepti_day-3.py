import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv("../input/machinelearning/50_Startups.csv")

X = dataset.iloc[ : , :-1].values

Y = dataset.iloc[ : ,  4 ].values
dataset


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelenc= LabelEncoder()

X[:,3] = labelenc.fit_transform(X[:, 3])



ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')

X = ct.fit_transform(X)


label_Y=LabelEncoder()

y=label_Y.fit_transform(Y)
# avoid dummy variable



X= X[: , 1:]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, random_state=0)

#fitting multiple linear regression to training set



from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(X_train, Y_train)
#Predicting the test results



y_pred=regressor.predict(X_test)
import matplotlib.pyplot as plt 

plt.plot(y_pred)
#---------------------DONE-----------------------------------#