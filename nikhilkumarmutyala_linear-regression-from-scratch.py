#import the required libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
#read the csv file into dataframe

df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
#getting to know about the datatypes

df.info()
#view dataframe

df

#finding the shape of dataframe

df.shape
#find the mean, count, min, max, percentile, standard deviation of given data

df.describe()
#checking for null values

df.isnull().sum()
#delete unnecessary column

del df['lat']
#view first 5 records

df.head()
#delete unnecessary column

del df['long']
#view first five records

df.head()
#delete unnecessary column

del df['view']
#view first 5 records

df.head()
#finding value counts of no of bedrooms 

df.bedrooms.value_counts()
#removing outlier with '33' bedrooms

df=df[df['bedrooms']!=33]
#check if the outlier is removed

df.bedrooms.value_counts()
#conevrting data column into float type

pd.options.display.float_format = '{:,.0f}'.format

df
#formatting the unncessary values in data column

df['date']=[x[:8] for x in df['date']]

df
#changing date column into date type using pandas

df['date']=pd.to_datetime(df["date"])

df
#checking the data types

df.info()
#replot for bedrooms and price

sns.set(style='white')

sns.relplot(x='bedrooms',y='price',hue='bedrooms',alpha=0.7,height=10,data=df)
#plot boxplot graph for bedrooms and price

sns.set(style='ticks',palette='pastel')

fig, ax = plt.subplots(figsize=(14,10)) 

sns.boxplot(x='bedrooms',y='price',palette=["m", "g"], data=df,ax=ax)
#boxplot between bathrooms and price

sns.set(style='ticks',palette='muted')

fig,ax=plt.subplots(figsize=(14,10))

sns.boxplot(x='bathrooms',y='price',palette=["b", "g"],data=df,ax=ax)
#find correlation

df.corr()
#heatmap for all the required features for correlation

df1=df[['price', 'bedrooms', 'bathrooms', 'sqft_living',

    'sqft_lot', 'floors', 'waterfront', 'condition', 'grade',

    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

     'sqft_living15', 'sqft_lot15']]

fig, ax = plt.subplots(figsize=(20,20)) 

sns.heatmap(df.corr(),cmap = 'YlGnBu',annot=True,ax=ax)
#graph for dates,price and bedrooms

fig,ax=plt.subplots(figsize=(14,10))

df.set_index('date').groupby('bedrooms')['price'].plot(legend=True)
#lineplot for sqft living and price

fig,ax=plt.subplots(figsize=(12,8))

palette=sns.color_palette('mako_r',6)

ax=sns.lineplot(x='sqft_living',y='price',data=df,hue='bedrooms',palette='mako_r')
#lineplot for sqft lot and price

fig,ax=plt.subplots(figsize=(12,8))

palette=sns.color_palette('mako_r',6)

ax=sns.lineplot(x='sqft_lot',y='price',data=df,hue='bedrooms',palette='mako_r')
#value counts of condition of the house

df['condition'].value_counts()
#catplot for condition and price

sns.catplot(x='condition',y='price',data=df)
#3d plot

fig=plt.figure(figsize=(20,13))

ax=fig.add_subplot(2,2,1,projection='3d')

ax.scatter(df['floors'],df['bedrooms'],df['bathrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nfloors',ylabel='\nbedrooms',zlabel='\nbathrooms')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,2,projection='3d')

ax.scatter(df['floors'],df['bedrooms'],df['sqft_living'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nfloors',ylabel='\nbedrooms',zlabel='\nsqft living')

ax.set(ylim=[0,12])



ax=fig.add_subplot(2,2,3,projection='3d')

ax.scatter(df['sqft_living'],df['sqft_lot'],df['bathrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nsqft living',ylabel='\nsqft lot',zlabel='\nbathrooms')

ax.set(ylim=[0,250000])



ax=fig.add_subplot(2,2,4,projection='3d')

ax.scatter(df['sqft_living'],df['sqft_lot'],df['bedrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nsqft living',ylabel='\nsqft lot',zlabel='\nbedrooms')

ax.set(ylim=[0,250000])
from abc import ABC, abstractmethod



# Super class for machine learning models 



class BaseModel(ABC):

    """ Super class for ITCS Machine Learning Class"""

    

    @abstractmethod

    def train(self, X, T):

        pass



    @abstractmethod

    def use(self, X):

        pass



    

class LinearModel(BaseModel):

    """

        Abstract class for a linear model 

        

        Attributes

        ==========

        w       ndarray

                weight vector/matrix

    """



    def __init__(self):

        """

            weight vector w is initialized as None

        """

        self.w = None



    # check if the matrix is 2-dimensional. if not, raise an exception    

    def _check_matrix(self, mat, name):

        if len(mat.shape) != 2:

            raise ValueError(''.join(["Wrong matrix ", name]))

        

    # add a basis

    def add_ones(self, X):

        """

            add a column basis to X input matrix

        """

        self._check_matrix(X, 'X')

        return np.hstack((np.ones((X.shape[0], 1)), X))



    ####################################################

    #### abstract funcitons ############################

    @abstractmethod

    def train(self, X, T):

        """

            train linear model

            

            parameters

            -----------

            X     2d array

                  input data

            T     2d array

                  target labels

        """        

        pass

    

    @abstractmethod

    def use(self, X):

        """

            apply the learned model to input X

            

            parameters

            ----------

            X     2d array

                  input data

            

        """        

        pass
# Linear Regression Class for least squares

class LinearRegress(LinearModel): 

    """ 

        LinearRegress class 

        

        attributes

        ===========

        w    nd.array  (column vector/matrix)

             weights

    """

    def __init__(self):

        LinearModel.__init__(self)

        

    # train lease-squares model

    def train(self, X, T):

        X = self.add_ones(X)

        xtrans = X.T.dot(X)

        self.w = np.linalg.pinv(xtrans).dot(X.T).dot(T)

        self.w = self.w.T

        print(self.w)

        return self.w.T

    

    # apply the learned model to data X

    def use(self, X):

        X = self.add_ones(X)

        a=self.w.dot(X.T)

        return a.T

       
import collections # for checking iterable instance



# LMS class 

class LMS(LinearModel):

    """

        Lease Mean Squares. online learning algorithm

    

        attributes

        ==========

        w        nd.array

                 weight matrix

        alpha    float

                 learning rate

    """

    def __init__(self, alpha):

        LinearModel.__init__(self)

        self.alpha = alpha

    

    # batch training by using train_step function

    def train(self, X, T):

        for x,t in zip(X,T):

            self.train_step(x,t)

            

    # train LMS model one step 

    # here the x is 1d vector

    def train_step(self, x, t):

        x = x.reshape(1,x.size)

        xr = self.add_ones(x)

        t = t.reshape(t.size,1)

        if self.w is None:

            self.w = np.zeros((xr.shape[1],1))

        self.w = self.w - self.alpha*(xr@self.w - t)*xr.T     ## TODO: replace this with your codes

        #else

            #self.w= self.train_step(x,t)

    

    # apply the current model to data X

    def use(self, X):

        N = X.shape[0]

        X1 = np.hstack((np.ones((N, 1)), X.reshape((X.shape[0], -1))))

        return X1 @ self.w  

        
import matplotlib.pyplot as plt

%matplotlib inline
X = np.linspace(0,10, 11).reshape((-1, 1))

T = -2 * X + 3.2

ls = LinearRegress()

ls.train(X, T)

plt.plot(ls.use(X))
fig,ax=plt.subplots(figsize=(8,8))

X = np.linspace(0,10, 11).reshape((-1, 1))

T = -2 * X + 3.2

ls = LinearRegress()

ls.train(X, T)

plt.plot(ls.use(X))
fig,ax=plt.subplots(figsize=(8,8))

lms = LMS(0.01)

for x, t in zip(X, T):

    lms.train_step(x, t)

    plt.plot(lms.use(X))
lms.train(X, T)

plt.plot(lms.use(X))
#We create X and y and find their values. Now y is an array and it is transposed.

X=df[['bedrooms','bathrooms','sqft_living','sqft_lot','condition','grade','floors']].values

y=df['price'].values

y=np.array([y])

y=y.transpose()
#We are plotting graph for train and prediction of the data for least squares

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=15)

ls=LinearRegress()

ls.train(X_train,y_train)

y_pred=ls.use(X_test)

fig,ax=plt.subplots(figsize=(10,7))

plt.scatter(y_test,y_pred)

plt.plot(X_test,ls.use(X_test))  
#plot the graph for X training set 

fig,ax=plt.subplots(figsize=(10,7))

plt.plot(ls.use(X_train))
#we plot graph for difference between y test and y prediction

fig,ax=plt.subplots(figsize=(10,7))

sns.distplot((y_test-y_pred),bins=20);
#plot graph for y test and y prediction to see the variation of the data

fig,ax=plt.subplots(figsize=(10,7))

plt.plot(y_test, 'g-*')

plt.plot(y_pred, 'r-+')
##################### WHAT I WILL RELEASE ############



# Self-Test code for accuracy of your model - DO NOT MODIFY THIS

# Primilnary test data

X = np.array([[2,5],

              [6,2],

              [1,9],

              [4,5],

              [6,3],

              [7,4],

              [8,3]])

T = X[:,0, None] * 3 - 2 * X[:, 1, None] + 3

N = X.shape[0]



def rmse(T, Y):

    return np.sqrt(np.sum((T-Y)**2))



model_names = ['LS', 'LMS_All', 'LMS_1STEP']

models = [LinearRegress(), LMS(0.02), LMS(0.02)]

#train

for i, model in enumerate(models):

    print("training ", model_names[i], "...") 

    if i == len(models) -1: 

        # train only one step for LMS2

        model.train_step(X[0], T[0])

    else:

        model.train(X, T)



def check(a, b, eps=np.finfo(float).eps):

    if abs(a-b) > eps:

        print("failed.", a, b)

    else:

        print("passed.")



errors = [1.19e-13, 2.8753214702, 38.0584918251]

for i, model in enumerate(models):

    print("---- Testing ", model_names[i], "...", end=" ") 

    

    # rmse test

    err = rmse(T, model.use(X))

    if check(err, errors[i], eps=1e-10):

        print ("check your weights: ", model.w)

        print ("oracle: ", )