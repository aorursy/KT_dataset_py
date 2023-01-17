# Import Library

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Image
# Reading the data file

data= pd.read_csv('../input/tvradionewspaperadvertising/Advertising.csv') 
data.head() # Display first five rows from the dataset
data.shape
data.info() #summary of the dataframe
data.isna().sum() # finding the count of missing values
# visualize the relationship between the features and the response using scatterplots

fig, axs = plt.subplots(1, 3, sharey=True)

data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(10, 8))

data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])

data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
# create X and y

feature_cols = ['TV']

X = data[feature_cols]

y = data.Sales



# follow the usual sklearn pattern: import, instantiate, fit

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X, y)



# print intercept and coefficients

print(lm.intercept_)

print(lm.coef_)
#calculate the prediction

6.974821 + 0.0554647*50
#  Let's create a DataFrame since the model expects it

X_new = pd.DataFrame({'TV': [50]})

X_new.head()
# use the model to make predictions on a new value

lm.predict(X_new)
# create a DataFrame with the minimum and maximum values of TV

X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})

X_new.head()
# make predictions for those x values and store them

preds = lm.predict(X_new)

preds
# first, plot the observed data

data.plot(kind='scatter', x='TV', y='Sales')



# then, plot the least squares line

plt.plot(X_new, preds, c='red', linewidth=2)
#constant is automatically added to your data and intercept id fitted whereas in statsmodelsapi we have to add constant

import statsmodels.formula.api as smf

lm = smf.ols(formula='Sales ~ TV', data=data).fit()

lm.conf_int()



#a = lm.summary()

#print(a)
# print the p-values for the model coefficients

lm.pvalues
# print the R-squared value for the model

lm.rsquared
# create X and y

feature_cols = ['TV', 'Radio', 'Newspaper']

X = data[feature_cols]

y = data.Sales



lm = LinearRegression()

lm.fit(X, y)



# print intercept and coefficients

print(lm.intercept_)

print(lm.coef_)
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

lm.conf_int()

lm.summary()
# only include TV and Radio in the model

lm = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()

lm.rsquared
# add Newspaper to the model (which we believe has no association with Sales)

lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

lm.rsquared
import numpy as np



# set a seed for reproducibility

np.random.seed(12345)



# create a Series of booleans in which roughly half are True

nums = np.random.rand(len(data))

mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large

data['Scale'] = 'small'

data.loc[mask_large, 'Scale'] = 'large'

data.head()
# create a new Series called IsLarge

data['IsLarge'] = data.Scale.map({'small':0, 'large':1})

data.head()
# create X and y

feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge']

X = data[feature_cols]

y = data.Sales



# instantiate, fit

lm = LinearRegression()

lm.fit(X, y)



# print coefficients

i=0

for col in feature_cols:

    print('The Coefficient of ',col, ' is: ',lm.coef_[i])

    i=i+1
# set a seed for reproducibility

np.random.seed(123456)



# assign roughly one third of observations to each group

nums = np.random.rand(len(data))

mask_suburban = (nums > 0.33) & (nums < 0.66)

mask_urban = nums > 0.66

data['Targeted Geography'] = 'rural'

data.loc[mask_suburban, 'Targeted Geography'] = 'suburban'

data.loc[mask_urban, 'Targeted Geography'] = 'urban'

data.head()
# create three dummy variables using get_dummies, then exclude the first dummy column

area_dummies = pd.get_dummies(data['Targeted Geography'], prefix='Targeted Geography').iloc[:, 1:]



# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)

data = pd.concat([data, area_dummies], axis=1)

data.head()
# create X and y

feature_cols = ['TV', 'Radio', 'Newspaper', 'IsLarge', 'Targeted Geography_suburban', 'Targeted Geography_urban']

X = data[feature_cols]

y = data.Sales



# instantiate, fit

lm = LinearRegression()

lm.fit(X, y)



# print coefficients

print(feature_cols, lm.coef_)
#Let's start with importing necessary libraries



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
data =pd.read_csv('../input/admission-prediction/Admission_Prediction.csv')

data.head()
data.describe(include='all')
data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])

data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())

data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())
data.describe()
data= data.drop(columns = ['Serial No.'])

data.head()
# let's see how data is distributed for every column

plt.figure(figsize=(10,10), facecolor='white')

plotnumber = 1



for column in data:

    if plotnumber<=16 :

        ax = plt.subplot(4,4,plotnumber)

        sns.distplot(data[column])

        plt.xlabel(column,fontsize=10)

    plotnumber+=1

plt.tight_layout()
y = data['Chance of Admit']

X =data.drop(columns = ['Chance of Admit'])
plt.figure(figsize=(10,10), facecolor='white')

plotnumber = 1



for column in X:

    if plotnumber<=15 :

        ax = plt.subplot(5,3,plotnumber)

        plt.scatter(X[column],y)

        plt.xlabel(column,fontsize=10)

        plt.ylabel('Chance of Admit',fontsize=10)

    plotnumber+=1

plt.tight_layout()
scaler =StandardScaler()



X_scaled = scaler.fit_transform(X)
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
y_train
regression = LinearRegression()



regression.fit(x_train,y_train)
regression.score(x_train,y_train)
adj_r2(x_train,y_train)
regression.score(x_test,y_test)
adj_r2(x_test,y_test)
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
lasso_reg.score(x_test, y_test)
# Using Ridge regression model

# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 

# We will pass an array of random numbers for ridgeCV to select best alpha from them



alphas = np.random.uniform(low=0, high=10, size=(50,))

ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)

ridgecv.fit(x_train, y_train)
ridgecv.alpha_
ridge_model = Ridge(alpha=ridgecv.alpha_)

ridge_model.fit(x_train, y_train)
ridge_model.score(x_test, y_test)
# Elastic net



elasticCV = ElasticNetCV(alphas = None, cv =10)



elasticCV.fit(x_train, y_train)
elasticCV.alpha_
# l1_ration gives how close the model is to L1 regularization, below value indicates we are giving equal

#preference to L1 and L2

elasticCV.l1_ratio
elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)

elasticnet_reg.fit(x_train, y_train)
elasticnet_reg.score(x_test, y_test)
# saving the model to the local file system

import pickle

filename = 'LR.pickle'

pickle.dump(regression, open(filename, 'wb'))