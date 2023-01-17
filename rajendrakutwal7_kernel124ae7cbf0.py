# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn import linear_model,metrics
from sklearn.metrics import accuracy_score
import csv

x_file = open(("../input")+"/train.csv", "r") 
y_file = open(("../input")+"/test.csv", "r") 

df = pd.read_csv(x_file)
df1= pd.read_csv(y_file)

#print(df.head(10))

print(df.columns)
#print(df1.columns)
X_Check=df['YearBuilt']
X=df[['YearBuilt','OverallCond','YrSold','LotArea','ScreenPorch','1stFlrSF','2ndFlrSF','Fireplaces','PoolArea']]
Y=df['SalePrice']
X_test=df1[['YearBuilt','OverallCond','YrSold','LotArea','ScreenPorch','1stFlrSF','2ndFlrSF','Fireplaces','PoolArea']]
Y.astype(np.float64)
#X.head()

reg = linear_model.LinearRegression()
reg.fit(X,Y)
pred=reg.predict(X_test)
results = model_selection.cross_val_score(reg, X_test,pred)
b, m = polyfit(X_Check,Y, 1)
plt.plot(X_Check,Y, '.')
plt.plot(X_Check, b + m * X_Check, '*')
plt.show()


#Accuracy

#print('Accuracy: \n', results)
# regression coefficients
#print('Prediction: \n', Y)
#print('Coefficients: \n', reg.coef_)
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test,pred)))

#sns.barplot(X,Y)
#plt.title("Chart For Price")
#plt.xlabel("Ratings")
#plt.ylabel("Price")
#plt.show()






# Any results you write to the current directory are saved as output.

