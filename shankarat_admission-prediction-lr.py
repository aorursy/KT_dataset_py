import pandas as pd 

import numpy as np 

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression

from sklearn.model_selection import train_test_split

import statsmodels.api as sm 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Let's create a function to create adjusted R-Squared

def adj_r2(x,y):

    r2 = regression.score(x,y)

    n = x.shape[0]

    p = x.shape[1]

    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    return adjusted_r2
# You may need to run below script to find your file location and use the path to fetch your data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read prediction data



data =pd.read_csv('/kaggle/input/Admission_Prediction.csv')

data.head()
data.describe(include='all')
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])

data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())

data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())
# Drop SL number, which is not required.

data= data.drop(columns = ['Serial No.'])



#Let's visualize the data and analyze the relationship between independent and dependent variables:

plt.figure(figsize=(20,25), facecolor='white')

plotnumber = 1



for column in data:

    if plotnumber<=16 :

        ax = plt.subplot(4,4,plotnumber)

        sns.distplot(data[column])

        plt.xlabel(column,fontsize=20)

        #plt.ylabel('Salary',fontsize=20)

    plotnumber+=1

plt.tight_layout()
y = data['Chance of Admit']

X =data.drop(columns = ['Chance of Admit'])

plt.figure(figsize=(20,30), facecolor='white')

plotnumber = 1



for column in X:

    if plotnumber<=15 :

        ax = plt.subplot(5,3,plotnumber)

        plt.scatter(X[column],y)

        plt.xlabel(column,fontsize=20)

        plt.ylabel('Chance of Admit',fontsize=20)

    plotnumber+=1

plt.tight_layout()
# We don't want to see warnings in the kernel

import warnings

warnings.filterwarnings('ignore')



# Standard Scaler

scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)



# Import VIF to check multicollinearity on numarical data

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = X_scaled



# we create a new data frame which will include all the VIFs

# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)

# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do

vif = pd.DataFrame()



# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 

vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]

# Finally, I like to include names so it is easier to explore the result

vif["Features"] = X.columns

vif
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)



regression = LinearRegression()

regression.fit(x_train,y_train)
print ('regression.score on Train ===>',regression.score(x_train,y_train))

print ('adj_r2 on Train           ===>',adj_r2(x_train,y_train))

print ('regression.score on Test  ===>',regression.score(x_test,y_test))

print ('adj_r2 on Test            ===>', adj_r2(x_test,y_test))
# Lasso Regularization

# LassoCV will return best alpha and coefficients after performing 10 cross validations

lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)

lasscv.fit(x_train, y_train)



# best alpha parameter

alpha = lasscv.alpha_

alpha
#now that we have best parameter, let's use Lasso regression and see how well our data has fitted before



lasso_reg = Lasso(alpha)

lasso_reg.fit(x_train, y_train)



#check the score

lasso_reg.score(x_test, y_test)
# Using Ridge regression model

# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 

# We will pass an array of random numbers for ridgeCV to select best alpha from them



alphas = np.random.uniform(low=0, high=10, size=(50,))

ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)

ridgecv.fit(x_train, y_train)



ridgecv.alpha_
# Fit the model with new alpha and find the score

ridge_model = Ridge(alpha=ridgecv.alpha_)

ridge_model.fit(x_train, y_train)



ridge_model.score(x_test, y_test)
# Elastic net



elasticCV = ElasticNetCV(alphas = None, cv =10)

elasticCV.fit(x_train, y_train)

print ('elasticCV',elasticCV.alpha_)

print ('elasticCV Ration',elasticCV.l1_ratio)
elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)

elasticnet_reg.fit(x_train, y_train)



elasticnet_reg.score(x_test, y_test)