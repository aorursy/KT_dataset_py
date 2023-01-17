import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#file copy library

from shutil import copyfile



# copy our file into the working directory 

copyfile(src = "../input/fncspy/fncs.py", dst = "../working/fncspy.py") 



# import all of our custom functions

from fncspy import *



# Loading training, validation and test data

dfTrain = pd.read_csv('../input/hyperparameterselection/Data_Train.csv')

dfVal = pd.read_csv('../input/hyperparameterselection/Data_Val.csv')

dfTest = pd.read_csv('../input/hyperparameterselection/Data_Test.csv')



############ TRAINING A MODEL



# Fitting model

deg = 5

X = create_X(dfTrain.x,deg)

beta = fit_beta(dfTrain,deg)



# Computing training error

yPredTrain = predict_y(dfTrain.x,beta)

err = rmse(dfTrain.y,yPredTrain)

print('Training Error = {:2.3}'.format(err))



# Computing test error

yPredTest = predict_y(dfTest.x,beta)

err = rmse(dfTest.y,yPredTest)

print('Test Error = {:2.3}'.format(err))



# Plotting fitted model

x = np.linspace(0,1,100)

y = predict_y(x,beta)

plt.plot(x,y,'b-',dfTrain.x,dfTrain.y,'rs')

plt.legend(['Prediction','Training'])

plt.show()
# Computing error

maxDegree = 5

err = computeError(maxDegree,dfTrain,dfVal)



# Plotting error

plotError(err)



# Selecting optimal degree

degOpt = err['deg'][np.argmin(err['errVal'])]

print('Optimal Degree = {:1}'.format(degOpt))
# Fitting model with training data only

beta = fit_beta(dfTrain,degOpt)

errTest = rmse(dfTest.y,predict_y(dfTest.x,beta))

print('Test Error using only training data = {:2.3}'.format(errTest))



# Fitting model with training and val data

df = pandas.concat([dfTrain, dfVal])

beta = fit_beta(df,degOpt)

errTest = rmse(dfTest.y,predict_y(dfTest.x,beta))

print('Test Error using train. & val. data = {:2.3}'.format(errTest))



# Plotting prediction

x = np.linspace(0,1,100)

y = predict_y(x,beta)

plt.plot(x,y,'b-',df.x,df.y,'rs',dfTest.x,dfTest.y,'k.')

plt.legend(['Prediction','Train+Val','Test'])

plt.show()