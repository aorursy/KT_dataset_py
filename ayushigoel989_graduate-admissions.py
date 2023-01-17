import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

os.listdir("../input")

dataset=pd.read_csv("../input/Admission_Predict.csv")

dataset.head()

dataset=dataset.drop("Serial No.", axis=1)
X=dataset.iloc[: , :-1].values

y=dataset.iloc[: , -1].values

y=y.reshape(-1,1)


pd.scatter_matrix(dataset,alpha=0.6)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)

print(y_pred)


lin_reg.score(X_train, y_train)



lin_reg.score(X_test, y_test)
lin_reg.score(X, y)


lin_reg.coef_