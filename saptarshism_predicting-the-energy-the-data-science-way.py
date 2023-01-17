import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/roboBohr.csv',index_col=0)
data.head()
data.drop('pubchem_id',axis=1,inplace=True)
data.head()
data.isnull().sum().sum()
import matplotlib.pyplot as plt

import seaborn as sns
fig = plt.figure(figsize=(8,6))

sns.distplot(data['Eat'],bins=50)

plt.title("Atomic Energy Distribution")

plt.xlabel("Atomic Energy (Eat)")
X = data.drop('Eat',axis=1).as_matrix()

y = data['Eat'].as_matrix()
from sklearn.linear_model import LinearRegression,BayesianRidge

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

linear_model = LinearRegression()

linear_model.fit(X_train,y_train)

pred = linear_model.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error

print("The MAE is {}.\nThe MSE is {}".format(mean_absolute_error(y_test,pred),mean_squared_error(y_test,pred)))

fig = plt.figure(figsize=(6,3))

sns.distplot((pred-y_test),kde=False)

plt.title("Linear Regression Error Distribution")

plt.xlabel("Error Value")
Bayesian_model = BayesianRidge()

Bayesian_model.fit(X_train,y_train)

pred = Bayesian_model.predict(X_test)

print("The MAE is {}.\nThe MSE is {}".format(mean_absolute_error(y_test,pred),mean_squared_error(y_test,pred)))

print("The Cross Validation Scores are: {}".format(cross_val_score(Bayesian_model,X,y,cv=10)))

fig = plt.figure(figsize=(6,3))

sns.distplot((pred-y_test),kde=False)

plt.title("Bayesian Ridge Regression Error Distribution")

plt.xlabel("Error Value")
from sklearn.decomposition import PCA