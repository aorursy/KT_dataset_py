import pandas as pd

import math 

import numpy as np
tr = pd.read_csv("/kaggle/input/random-linear-regression/train.csv")

tst = pd.read_csv("/kaggle/input/random-linear-regression/test.csv")
tst
tr['y'].dropna()
trn = tr.dropna()

trn
import matplotlib.pyplot as plt





%matplotlib inline
trn.describe()
plt.title('Relationship between X and Y')

plt.scatter(trn['x'], trn['y'],  color='black')

plt.show()
from sklearn import linear_model
lm = linear_model.LinearRegression()

trnX = np.array(trn['x']).reshape(-1,1)

trnY = trn['y']

lm.fit(trnX,trnY)

# This the coefficient of the feature

print("Coefficient for X : ", lm.coef_)



# Let's look at R sq to give an idea of the fit 

print('R sq              : ',lm.score(trnX,trnY))



# correlation

print('Correlation       : ', math.sqrt(lm.score(trnX,trnY)))



# good R sq and correlation!
tstX = np.array(tst['x']).reshape(-1,1)

y_pr = lm.predict(tstX)

plt.title('Comparison of Y values in test and the Predicted values')

plt.ylabel('Test Set')

plt.xlabel('Predicted values')

plt.scatter(y_pr, tst['y'] ,  color='black')

plt.show()
plt.title('try')

plt.scatter(tst['x'],tst['y'],color='g')

plt.scatter(tst['x'],y_pr,color='r')



plt.show()
from sklearn.metrics import r2_score

r2_score(tst['y'],y_pr)