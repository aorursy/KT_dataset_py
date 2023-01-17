# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col="Serial No.")

print(df)
# df_new = df[['CGPA','GRE Score','TOEFL Score','University Rating','SOP']]

df_new = df

y = df.iloc[:,-1]

# y = y**2;

X_train = np.array(df_new.iloc[0:300,:-1])

X_val = np.array(df_new.iloc[300:400,:-1])

X_test = np.array(df_new.iloc[400:500,:-1])



y_train = np.array(y[0:300])

y_val = np.array(y[300:400])

y_test = np.array(y[400:500])

print(X_train.shape,X_val.shape,X_test.shape,y_train.shape,y_test.shape,y_val.shape)
def NormalizeData(X):

	mean = np.mean(X,axis=0,keepdims=True)

	sigma = np.std(X,axis=0,keepdims=True)

	X_norm = (X-mean)/sigma;

	return [X_norm,mean,sigma]



def sgd_momentum(w, dw, learning_rate=None,config=None):



    if config is None: config = {}

    # config.setdefault('learning_rate', 1e-1)

    config.setdefault('momentum', 0.9)

    v = config.get('velocity', np.zeros_like(w))

    if learning_rate==None:

    	learning_rate = 1e-2





    next_w = None

    v  = config['momentum']*v - learning_rate*dw

    next_w = w + v

    config['velocity'] = v

    return next_w, config



def computeCost(X,y,theta,reg=None):

	if(reg==None):

		reg = 0



	m = X.shape[0]

	J = np.sum((X.dot(theta) - y)**2)/(2*m)

	J += (reg*np.sum(theta**2))/(2*m)

	dtheta = ((X.T).dot(  X.dot(theta) - y  ))/m

	dtheta += reg*np.sum(theta)/m

	return [J,dtheta];



def gradientDescentWithMomentum(X,y,theta,num_iters,reg=None,learning_rate=None,printEpochs=None):

	m = X.shape[0]

	J_history = np.zeros((num_iters, 1))

	dtheta = ((X.T).dot(  X.dot(theta) - y  ))/m



	for i in range(num_iters):

		theta,config = sgd_momentum(theta, dtheta,learning_rate, config=None)

		[J_history[i],dtheta] = computeCost(X,y,theta,reg);

		if(printEpochs==True):

			print("Iteration ", i, " : ", J_history[i])

		if(i>1 and abs(J_history[i]-J_history[i-1])<1e-5):

			break



	return [theta,J_history]

def LinearReg(X,y,num_iters,reg,learning_rate,theta,verbose):

  num_training  = X.shape[0]

  try:

    num_features  = X.shape[1]

  except:

    num_features = 1;



  X = X.reshape((num_training,num_features))

  y = y.reshape((num_training,1))

  [X,mean,sigma] = NormalizeData(X)



  X = np.hstack( (np.ones((num_training,1)),X ) )

  theta = np.random.rand(num_features+1,1) 

  [theta,J_history] = gradientDescentWithMomentum(X,y,theta,num_iters,reg,learning_rate,verbose);

  return [theta,J_history]

learning_rate = 1e-2

reg = 0.5

num_iters = 1000

theta = np.random.rand(X_train.shape[0]+1,1)

[theta,J_history] = LinearReg(X_train,y_train,num_iters,reg,learning_rate,theta,True)
test_reg = [0,0.00005,0.00025,0.0005,0.00075,0.001,0.005,0.025,0.05,0.10,0.25,0.5,0.75,0.90,1,1.1,1.4,1.6,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,5,]

best_reg = 0.5;

temp = np.hstack( (np.ones((X_val.shape[0],1)),X_val) )

[best_J,t] = computeCost(temp,y_val,theta,0.5);

best_theta=theta;

learning_rate = 1e-2

# print(X_val,y_val)

for i in test_reg:

  [theta_t,J_history] = LinearReg(X_val,y_val,num_iters,i,learning_rate,theta,False)

  temp = np.hstack( (np.ones((X_val.shape[0],1)),X_val ) )

  [J,dtheta] = computeCost(temp,y_val,theta_t,i)

  print(J)

  if(J<best_J):

    best_J = J

    best_reg = i

    best_theta = theta_t



print(best_reg,best_J)  



plt.plot(J_history)
temp = X_test

theta = best_theta

[temp,mean,sigma] = NormalizeData(temp)

temp = np.hstack( (np.ones((X_test.shape[0],1)),temp ) )

pred = temp.dot(theta)

y_test = y_test.reshape((100,1))

print(y_test.shape,pred.shape)

fin = np.hstack( (y_test,pred ) )

print(fin)

print("RMSE : ", np.sqrt(np.sum((y_test-pred)**2)/100))