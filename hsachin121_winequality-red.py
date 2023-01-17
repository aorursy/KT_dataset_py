import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../input/winequality-red.csv')
data.head(5)
data.groupby('quality').count()
data.isna().sum()
plt.figure(figsize=(8,7))

plt.scatter(data['pH'],data['quality'])

plt.xlabel('PH level')

plt.ylabel('Quality')
Y=data['quality']

mapping={3:'a',4:'b',5:'c',6:'d',7:'e',8:'f'}

data

X=data.drop(columns=['quality'])
clf = LinearRegression()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)
clf.fit(xtrain,ytrain)
clf.predict(xtest)
clf.score(xtest,ytest)