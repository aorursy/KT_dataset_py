import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
%matplotlib inline
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

dataDF = pd.read_csv('/kaggle/input/startup-logistic-regression/50_Startups.csv')
dataDF.info()

dataDF.head()

dataDF.tail()
sn.heatmap(dataDF.corr(),annot=True)
sn.pairplot(dataDF)
plt.scatter(dataDF.Administration,dataDF.Profit)
plt.xlabel('Administration Spend')
plt.ylabel('Profit')
plt.show()
plt.scatter(dataDF[['Marketing Spend']],dataDF.Profit)
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()
plt.scatter(dataDF[['R&D Spend']],dataDF.Profit)
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()
#### A model with only Marketing Spend and R&D Spend as features

x1 = dataDF[['Marketing Spend','R&D Spend']]
y1 = dataDF.Profit
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1,y1)
model1 = LinearRegression()
model1.fit(x1,y1)
y_predict1 = model1.predict(x_test1)
print('R2 VALUE'+ str(r2_score(y_test1,y_predict1)))
print('ROOT MEAN SQUARED ERROR '+str(np.sqrt(mean_squared_error(y_test1,y_predict1))))
x2 = dataDF[['Administration','R&D Spend','Marketing Spend']]
y2 = dataDF.Profit
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2,y2)
model2 = LinearRegression()
model2.fit(x_train2,y_train2)
y_predict2 = model2.predict(x_test2)
print('R2 VALUE '+str(r2_score(y_test2,y_predict2)))
print('ROOT MEAN SQUARED ERROR '+str(np.sqrt(mean_squared_error(y_test2,y_predict2))))