## This dataset has muti variables, the goal of the project is to do it numpy.

## And i have also done using Elimination method 

# this is second kernel, any comments is appreciatable

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 
Startups = pd.read_csv("../input/50_Startups.csv")
Startups.head() ## which deplays first 5 rows 
X = Startups.iloc[:, :-1]

Y = Startups.iloc[:, 4].values
X
Y
# Ploting different variable to see how they are correlated 

Startups.hist(bins=15, color='red', edgecolor='black', linewidth=1.0,

           xlabelsize=8, ylabelsize=8, grid=False)    

plt.tight_layout(rect=(0, 0, 1.2, 1.2)) 
#SNS heatmap to correlerate

# Correlation Matrix Heatmap

# The heatmap shows how each variable are correlated to another variable

sub_plot, ax = plt.subplots(figsize=(10, 6))

corr = Startups.corr()

hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',

                 linewidths=.05)

sub_plot.subplots_adjust(top=0.93)

t= sub_plot.suptitle('Startups Attributes Correlation Heatmap', fontsize=14)
 ##Pair-wise Scatter Plots

cols = ['R&D Spend', 'Administration', 'Marketing Spend']

pp = sns.pairplot(Startups[cols], size=1.8, aspect=1.8,

                  plot_kws=dict(edgecolor="k", linewidth=0.5),

                  diag_kind="kde", diag_kws=dict(shade=True))



fig = pp.fig 

fig.subplots_adjust(top=0.93, wspace=0.3)

t = fig.suptitle(' Pairwise Plots', fontsize=14)
sns.stripplot(x='Profit', y = 'R&D Spend', data = Startups)

## The plot between Profit and R&d spend clearly shows the plot depends mainly upon R&D



sns.stripplot(x = 'Profit', y = 'Administration', data = Startups)

sns.pointplot(x = 'Profit', y = 'Marketing Spend', data = Startups)

## Marketing doesnt actually provide more profit, they are varying like shown in the plot 
######################################################################################

                                ###EDA###

#######################################################################################
# Creating dummy variables since our dataste has one catogorical variable

pd.get_dummies(X)



dummy = pd.get_dummies(Startups['State'])
dummy
X = pd.concat([X,dummy], axis = 1)
X
#now im deleting the column State

X.__delitem__('State')
X
#Normalizing the dataframe  by mean normalization

X = (X - X.mean())/X.std()

Y = (Y -Y.mean())/Y.std()
X
Y
#  to avoid dummy variable trap, i have delete one column from the dummy created. we delete one colunm to getoff the redundancy

X = X.iloc[:, 0:5]

ones = np.ones([X.shape[0],1])



X = np.concatenate((ones,X),axis = 1)



theta = np.zeros([1,6])
##Setting Hyper parameters initially

alpha = 0.03

iterate = 1500



Y.shape

Y = Y.reshape(50,1)
Y.shape
def costfunction(X,Y,theta):

    formula = np.power(((X @ theta.T)-Y),2)

    

    return np.sum(formula)/(2 * len(X))



cost = costfunction(X,Y,theta) 

print(cost)
def gradientDescent(X,Y,theta,iterate,alpha):

    cost = np.zeros(iterate)

    for i in range(iterate):

        theta = theta - (alpha/len(X)) * np.sum(X *(theta + X @ theta.T - Y), axis=0)

        cost[i] = costfunction(X, Y, theta)

    

    return theta,cost

G, cost = gradientDescent(X,Y,theta,iterate,alpha)

print(G)
## Now we put back to the cost function 

finalcost = costfunction(X,Y,G)

print(finalcost)

##ploting the graph of the gradiant decent



fig, ax = plt.subplots()  

ax.plot(np.arange(iterate), cost, 'r')  

ax.set_xlabel('Iterations')  

ax.set_ylabel('Cost2')  

ax.set_title('Error vs. Training Epochs')
##multilinear regression using backward elimination method 

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X
X_mat = X[:, [1,2,3,4,5,6]]
X_mat
regressor_OLS = sm.OLS(endog = Y, exog = X_mat).fit()

regressor_OLS.summary()
# From the previous summary of OLS, check for highest P value. If your p > Significance value 0.05. then 

# in next iteration we remove the variables with highest p value.
X_mat = X[:, [1,2,3,4,6]] 

regressor_OLS = sm.OLS(endog = Y, exog = X_mat).fit()

regressor_OLS.summary()
X_mat = X[:, [1,2,3,4]]

regressor_OLS = sm.OLS(endog = Y, exog = X_mat).fit()

regressor_OLS.summary()

X_mat = X[:, [1,2,3]]

regressor_OLS = sm.OLS(endog = Y, exog = X_mat).fit()

regressor_OLS.summary()
X_mat = X[:, [1,2]]

regressor_OLS = sm.OLS(endog = Y, exog = X_mat).fit()       

regressor_OLS.summary()



##now you can see the P value is higher so stop here.
## From the ols model , the more significant variable to the output(profit) is R&D spend.

#Stats model is used to find the most significant varaible 