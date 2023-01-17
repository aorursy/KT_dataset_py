import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
df=load_iris()
df.keys()
plt.plot(df.data)
x=df.data

y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6,random_state=42)
reg=LogisticRegression()

reg.fit(x_train,y_train)
reg.predict(x_test)
reg.score(x_test,y_test)*100
print('The accuracy rate : ',reg.score(x_test,y_test)*100,('%'))