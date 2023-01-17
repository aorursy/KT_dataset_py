import numpy as np

import pandas as pd

from sklearn import linear_model

from pandas import DataFrame

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_excel('../input/companydetails/companysalary.xlsx')

df
X = df[['Experience', 'Salary']]

y = df['Year_joining']



regr = linear_model.LinearRegression()

regr.fit(X, y)
print(regr.coef_) 
Y_predict = regr.predict(X)

Y_predict
plt.scatter(df['Experience'], df['Salary'], color='red')

plt.title('Salary Vs Experience', fontsize=14)

plt.xlabel('Experience', fontsize=14)

plt.ylabel('Salary', fontsize=14)

plt.grid(True)

plt.show()
df.head()
X = df[['Experience','Salary']]

y = df['Year_joining']



X1 = sm.add_constant(X)

est = sm.OLS(y, X1).fit()



est.summary()
df_sub = df[['Experience','Salary']]

df_sub.corr()
corrmat = df_sub.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, vmax=1, square=True);

plt.show()
fig_size = plt.rcParams["figure.figsize"] 

fig_size[0]=16.0

fig_size[1]=8.0

df_sub.hist(bins=100)

plt.show()
my_data = pd.read_excel('../input/companydetails/companysalary.xlsx')
print(df.columns)
print(df.head(5))
pd.isna(df).any()
x = df[['Experience','Salary']].values

y = df['Year_joining'].values.reshape(-1,1)

m = len(y)

print(m)
#we need to normalize the features using mean normalization

my_data = (my_data - my_data.mean())/my_data.std()

my_data.head()
#setting the matrixes

X = my_data.iloc[:,0:2]

ones = np.ones([X.shape[0],1])

X = np.concatenate((ones,X),axis=1)



y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray

theta = np.zeros([1,3])



#set hyper parameters

alpha = 0.01

iters = 1000
#computecost

def computeCost(X,y,theta):

    tobesummed = np.power(((X @ theta.T)-y),2)

    return np.sum(tobesummed)/(2 * len(X))
#gradient descent

def gradientDescent(X,y,theta,iters,alpha):

    cost = np.zeros(iters)

    for i in range(iters):

        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)

        cost[i] = computeCost(X, y, theta)

    

    return theta,cost



#running the gd and cost function

g,cost = gradientDescent(X,y,theta,iters,alpha)

print(g)



finalCost = computeCost(X,y,g)

print(finalCost)
#plot the cost

fig, ax = plt.subplots()  

ax.plot(np.arange(iters), cost, 'b')  

ax.set_xlabel('Experience')  

ax.set_ylabel('Year_joining')  

ax.set_title('Experience vs. Year_joining') 