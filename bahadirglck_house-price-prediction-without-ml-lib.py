# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.model_selection import train_test_split # split train and test data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class ML_LinearRegression:

    """ Apply Linear Regression

        

        Args: 

            lr : float. learning rate 

            iterations : int. How many iteration for training 

            regularization : Boolen. Use L2 regularization for model training

            lambda_reg : float. Regularization parameter 

            

    """

    def __init__(self,lr = 0.5, iterations = 10000, regularization = False, lambda_reg = None ):

        self.lr = lr 

        self.iterations = iterations

        self.regularization = regularization

        self.lambda_reg = lambda_reg

        

    def fit(self,x,y):

        """ Fit the our model

        

            Args: 

                x : np.array, shape = [n_samples, n_features]. Training Data 

                y : np.array, shape = [n_samples, n_conclusion]. Target Values

                

            Returns: 

                self : object

        """

        

        self.cost_list = []

        self.theta = np.zeros((x.shape[1],1))   

        #self.theta_zero = np.zeros((1,1))

        m = x.shape[0]   # samples in the data

        name_list=[]     # for plot x-axis name

        

        for i in range(self.iterations):  # Feed forward

            

            h_pred = np.dot(x,self.theta)     

            error = h_pred - y 

            if self.regularization == False: 

                

                cost = 1/(2*m)*(np.sum((error ** 2)))

                gradient_vector = np.dot(x.T, error)

                self.theta -= (self.lr/m) * gradient_vector # Gradient Descent

            

            else:

                cost = 1/(2*m)*(np.sum((error ** 2))) + (self.lambda_reg * (np.sum((self.theta ** 2 ))))  # add a L2 regularization

                gradient_vector = np.dot(x.T, error)

                self.theta -= (self.lr/m) * gradient_vector - ((self.lambda_reg/m) * self.theta) # Gradient Descent with L2 regularization

                

            self.cost_list.append(cost)

            name_list.append(i)

        

        plt.scatter(name_list,self.cost_list)

    

        return self

    

    def predict(self, x):

        """ Predicts the value after the model has been trained.

        

            Args: 

                x: np.array, shape = [n_samples, n_features]. Training Data

                

            Returns: 

                Predicted value 

        """

        

        

        return np.dot(x,self.theta)
dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv") #read dataset from .csv file
dataset.head(10)
dataset.iloc[:,1]
dataset.info()
dropped_dataset = dataset.drop(['id','date','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built',

                               'yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'],axis=1) # drop some features
sorted_df = dropped_dataset.sort_values('price') # sorted data by prices for visualisation
dataset_x = sorted_df.drop(['price'],axis=1)

dataset_y = sorted_df.price.values
dataset_x.head()
plt.figure(figsize=(19,4))



plt.subplot(141)

plt.scatter(dataset_x['bedrooms'],dataset_y)

plt.subplot(142)

plt.scatter(dataset_x['bathrooms'],dataset_y)

plt.subplot(143)

plt.scatter(dataset_x['sqft_living'],dataset_y)

plt.subplot(144)

plt.scatter(dataset_x['sqft_lot'],dataset_y)



plt.show()
dataset_x = (dataset_x - np.min(dataset_x)) / (np.max(dataset_x) - np.min(dataset_x))
x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size= 0.2, random_state= 42)
y_train = y_train.reshape((len(y_train), 1))

print(x_train.shape, y_train.shape)
linear = ML_LinearRegression(iterations=1000)

linear.fit(x_train,y_train)
linear_with_reg = ML_LinearRegression(iterations=1000,regularization=True, lambda_reg= 0.001)

linear_with_reg.fit(x_train,y_train)
linear_with_reg = ML_LinearRegression(iterations=1000,regularization=True, lambda_reg= 0.01)

linear_with_reg.fit(x_train,y_train)
y_pred_mymodel = linear.predict(x = x_test)
y_pred_mymodel_reg = linear_with_reg.predict(x=x_test)
plt.figure(figsize=(20,4))



# Visualization of y_test

plt.subplot(141) 

plt.plot(list(range(len(y_test))), y_test) 



# Visualization of difference between y_test and y_pred_mymodel

plt.subplot(142)

plt.plot(list(range(len(y_test))), y_test, 'b')

plt.plot(list(range(len(y_pred_mymodel))), y_pred_mymodel, 'r')



# Visualization of difference between y_test and y_pred_mymodel_reg

plt.subplot(143)

plt.plot(list(range(len(y_test))), y_test, 'b')

plt.plot(list(range(len(y_pred_mymodel_reg))), y_pred_mymodel_reg, 'r')



# Visualization of difference between y_pred_mymodel and y_pred_mymodel_reg

plt.subplot(144)

plt.plot(list(range(len(y_pred_mymodel))), y_pred_mymodel, 'b')

plt.plot(list(range(len(y_pred_mymodel_reg))), y_pred_mymodel_reg, 'r')



plt.show()
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)
y_pred_sklearn = regression.predict(x_test)
plt.plot(list(range(len(y_test))), y_test, 'b')

plt.plot(list(range(len(y_pred_sklearn))), y_pred_sklearn, 'r')

plt.show()
plt.plot(list(range(len(y_pred_sklearn))), y_pred_sklearn, 'b')

plt.plot(list(range(len(y_pred_mymodel))), y_pred_mymodel, 'r')

plt.show()
plt.plot(list(range(len(y_pred_sklearn))), y_pred_sklearn, 'b')

plt.plot(list(range(len(y_pred_mymodel_reg))), y_pred_mymodel_reg, 'r')

plt.show()