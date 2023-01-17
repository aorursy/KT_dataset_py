import pandas as pd # Reading csv file

import numpy as np # Linear algebra

from matplotlib import pyplot as plt # Graphing

from sklearn.model_selection import train_test_split # We will do a simple train, validate, test split.

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def ratioSMAPE(r):

    return np.abs(1 - r) / (1 + np.abs(r))



rvals = np.arange(0, 10, 0.1)

plt.plot(rvals, ratioSMAPE(rvals))

plt.title('Graph of |1 - r| / (1 + |r|)')

plt.show()
all_df = pd.read_csv('../input/train_1.csv')

all_df.shape
# Set all NaN to 0.



all_df.fillna(0, inplace = True)
# Separate into training data into features and targets.



futureT = 64

X_all = all_df.drop('Page', axis = 1).values[:, :-futureT]

Y_all = all_df.drop('Page', axis = 1).values[:, -futureT:]
# First split into test and a combination of training and validation.



X_trainvalid, X_test, Y_trainvalid, Y_test = train_test_split(X_all, Y_all, test_size = 0.33, random_state = 32)



# Now split up the training and validation sets.



X_train, X_valid, Y_train, Y_valid = train_test_split(X_trainvalid, Y_trainvalid, test_size = 0.33, random_state = 35)



print('X_train.shape = ', X_train.shape, '\tX_valid.shape = ', X_valid.shape, '\tX_test.shape = ', X_test.shape)

print('Y_train.shape = ', Y_train.shape, '\tY_valid.shape = ', Y_valid.shape, '\tY_test.shape = ', Y_test.shape)
def smape(Y_predict, Y_test):

    result = np.linalg.norm(Y_predict - Y_test, axis = 1)

    result = np.abs(result)

    denom = np.linalg.norm(Y_predict, axis = 1)

    denom += np.linalg.norm(Y_test, axis = 1)

    result /= denom

    result *= 100 * 2

    result = np.mean(result)

    return result
model = Pipeline([ ('means', FunctionTransformer(lambda X : X.mean(axis = 1, keepdims = True))),

                   ('tree', DecisionTreeRegressor(max_depth = 10)) ])

model.fit(X_trainvalid, Y_trainvalid)

Y_predict = model.predict(X_test)

test_smape = {} # We will be storing all of the results of our tests for comparison in a dictionary.

test_smape['benchmark'] = smape(Y_predict, Y_test)

print('SMAPE = ', test_smape['benchmark'])
# definition of ztransformation.



def ztransform1(Y, param):

    return 1 / (param + Y)



# inverse transformation, Y = inverseZ(Z)



def inverseZ1(Z, param):

    return -param + 1 / Z
# Values to try for param.



param_search = np.arange(20, 420, 20)



# To record results of fits



smapes = []

epsilon = 1e-6



for param in param_search:

    Z_train = ztransform1(Y_train, param)

    model.fit(X_train, Z_train)

    Z_predict = model.predict(X_valid)

    Y_predict = inverseZ1(Z_predict, param)

    newsmape = smape(epsilon + Y_predict, Y_valid)

    smapes.append(newsmape)

    print('param = ', param, ' smape = ', newsmape, ',\t', end = '')

    

plt.plot(param_search, smapes)

plt.title('SMAPES vs param values for Regression on ztransform1')

plt.show()

    
param = 240

Z_trainvalid = ztransform1(Y_trainvalid, param)

model.fit(X_trainvalid, Z_trainvalid)

Z_predict = model.predict(X_test)

Y_predict = inverseZ1(Z_predict, param)

test_smape['z1'] = smape(epsilon + Y_predict, Y_test)

print('SMAPE = ', test_smape['z1'])
def ztransform2(Y, param):

    return (np.sqrt(Y) - np.sqrt(param)) / np.sqrt(Y + param)



def inverseZ2(Z, param):

    Z2 = np.minimum(Z, 1 - epsilon)

    Z2 = np.maximum(Z2, -1 + epsilon)

    result = -1 - Z2 * np.sqrt(2 - Z2**2)

    result = result / (Z2**2 - 1)

    result = param * result**2

    return result



# Plot ztransform2 for param = 2.0.



plt.figure(figsize = (15, 5))

plt.subplot(121)

domain = np.arange(0, 10, 0.1)

plt.plot(domain, ztransform2(domain, 2.0))

plt.title('ztransform2 for param = 2.0')



# Graph inverseZ2 for param = 1.0 and reflection of graph of ztransform2.



plt.subplot(122)

domain2 = np.arange(-0.7, 0.7, 0.1)

plt.plot(domain2, inverseZ2(domain2, 2.0))

plt.plot(ztransform2(domain, 2.0), domain, color = 'red')

plt.title('Graph of inverseZ2 and reflection of graph of ztransform2')

plt.legend(['inverseZ2','reflection ztransform2'])

plt.show()
# Values to try for param.



param_search = np.arange(100, 2500, 100)



# To record results of fits



smapes = []

epsilon = 1e-6



for param in param_search:

    Z_train = ztransform2(Y_train, param)

    model.fit(X_train, Z_train)

    Z_predict = model.predict(X_valid)

    Y_predict = inverseZ2(Z_predict, param)

    newsmape = smape(epsilon + Y_predict, Y_valid)

    smapes.append(newsmape)

    print('param = ', param, ' smape = ', newsmape, ',\t', end = '')

    

plt.plot(param_search, smapes)

plt.title('SMAPES vs param values for Regression on ztransform2')

plt.show()

    
param = 1000

Z_trainvalid = ztransform2(Y_trainvalid, param)

model.fit(X_trainvalid, Z_trainvalid)

Z_predict = model.predict(X_test)

Y_predict = inverseZ2(Z_predict, param)

test_smape['z2'] = smape(epsilon + Y_predict, Y_test)

print('SMAPE = ', test_smape['z2'])
# Code to generate simple table using notebook output.



description = ['None (Benchmark)', 'Z Transformation 1', 'Z Transformation 2']

keys = ['benchmark', 'z1', 'z2']

scores = [test_smape[key] for key in keys]

output = [x for x in zip(description, scores)]



output