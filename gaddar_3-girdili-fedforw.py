# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv( '../input/qtest1test2train1000.csv' )
deney=data.astype('float')
#deney.convert_objects(convert_numeric=True)


test1=deney.iloc[0:1000,:]
test1_input=test1.iloc[:,0:3]
test1_output=test1.iloc[:,3]



test2=deney.iloc[1000:2000,:]
test2_input=test2.iloc[:,0:3]
test2_output=test2.iloc[:,3]


train=deney.iloc[2000:3000,:]
train_input=train.iloc[:,0:3]
train_output=train.iloc[:,3]

from sklearn.neural_network import MLPRegressor
my_network=MLPRegressor(hidden_layer_sizes=((2,3)),  
                        max_iter=500, random_state=42)
#solver : {‘lbfgs’, ‘sgd’, ‘adam’}
my_network.fit(train_input , train_output)

print('training data score:')
print (my_network.score(train_input , train_output))

print('test1 data score:')
print (my_network.score(test1_input , test1_output))

print('test2 data score:')
print (my_network.score(test2_input , test2_output))





#print("test accuracy: {} ".format(logreg.fit(train_input , train_output).score(test1_input , test1_output)))
#print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))