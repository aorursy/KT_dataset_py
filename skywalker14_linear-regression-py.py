import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.metrics import r2_score

dtrain = pd.read_csv('../input/train.csv')

dtest = pd.read_csv('../input/test.csv')



train=dtrain.dropna()

test=dtest.dropna()



train_x=np.array(train['x'])

train_y=np.array(train['y'])



test_x=np.array(test['x'])

test_y=np.array(test['y'])



#plt.scatter(train_x,train_y)

plt.scatter(test_x,test_y)



slope,intercept,r_value,p_value,std_err=stats.linregress(train_x,train_y)



def predict_train(X):

	Z=np.zeros(699)

	for i in range(len(train_x)):

		Z[i]=slope*X[i]+intercept

	return Z		

fitline=predict_train(train_x)

plt.plot(train_x,fitline,c='r')

def predict_test(X):

	Z=np.zeros(300)

	for i in range(len(test_x)):

		Z[i]=slope*X[i]+intercept

	return Z		

r2=r2_score(test_y,predict_test(test_x))

print ("R2 score is:  ")

print (r2)

plt.show()