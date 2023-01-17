# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra|

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


sample_size = 30

x1 = np.random.uniform(0,10,sample_size)

x2 = np.random.uniform(0,10,sample_size)

x3 = np.random.uniform(0,10,sample_size)



x0  = np.ones(sample_size)



pm_notT =np.array([x0,x1,x2,x3])

predictor_matrix = pm_notT.T

regression_coefficent = np.array([1,2,3,5])

real_value = np.add.reduce(predictor_matrix,axis = 1) + np.random.uniform(0,10,sample_size)

prediction = real_value
print('shape of predictor matrix',predictor_matrix.shape,'\n',

     'shape of the prediction matrix', prediction.shape , '\n',

     'shape of the regression coefficet matrix', regression_coefficent.shape)
learning_rate = 0.001

number_of_iterations = 1000
for i in range(number_of_iterations):



    regression_coefficent = regression_coefficent - (learning_rate*(np.dot(

                                                                            np.subtract(

                                                                                    np.dot(predictor_matrix ,regression_coefficent)

                                                                                    ,real_value ),

                                                                            predictor_matrix))/sample_size)

    

    mse = np.sum(np.square(

                        np.subtract(np.dot(predictor_matrix ,regression_coefficent),real_value )))/sample_size



    print(f'new rgc is {regression_coefficent} and mse is {mse}.')

                                                            
dataframe = pd.DataFrame(predictor_matrix,columns = ['variable_1','variable_2','variable_3','variable_4'])

dataframe['prediction'] = prediction

dataframe.head()
plt.figure()

plt.scatter(dataframe.variable_2,dataframe.prediction)