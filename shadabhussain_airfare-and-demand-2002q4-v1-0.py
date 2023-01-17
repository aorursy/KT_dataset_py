# importing packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# for expanding dataframe and displaying all columns

pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)
# loading data from dat file to a dataframe

df = pd.read_csv('../input/airfare-and-demand-2002/airq402.dat', sep='\s+', header=None)
# displaying top 5 rows and we can observe we dont have column labels

df.head()
# renaming columns

df.columns = ['City1', 'City2', 'Average_Fare0', 'Distance', 'Average Weekly Passengers', 'Market Leading Airline', 

           'Market_Share1', 'Average_Fare1', 'Low Price Airline', 'Market_Share2', 'Average_Fare2']
# top 5 rows after renaming columns

df.head()
df.info()
df.describe()
# pair plot function

def plot_pair(df):

    fig=plt.figure(figsize=(64,64))

    sns.pairplot(df)

    plt.show()
# box plot function

def plot_box(df):

    plt.figure(figsize=(25, 30))

    i=1

    for each in columns:

        plt.subplot(3, 3, i)

        sns.boxplot(y = each,data = df)

        i+=1

    plt.show()
# Correlation plot function

def plot_corr(df):

    plt.figure(figsize=(20, 14))

    sns.heatmap(df.corr(), cmap='YlGnBu', annot = True)

    plt.show()
# # Scatter plot function

# def plot_scatter(X,y):

#   plt.figure(figsize=(25, 30))

#   i=1

#   for each in [2,3,5,6,8,9]:

#     plt.subplot(3,2, i)

#     sns.scatter(X.iloc[:,each], y, alpha=0.5)

#     plt.show()
columns = ['Average_Fare0', 'Distance', 'Average Weekly Passengers', 'Market_Share1', 'Average_Fare1', 'Market_Share2', 'Average_Fare2']
plot_pair(df)

plt.show()
plot_corr(df)
plot_box(df)
leading_airline_group = df.groupby(['Market Leading Airline'])['City1'].size()

leading_airline_group.plot.bar(figsize=(18,8))

plt.show()
low_price_airline_group = df.groupby(['Low Price Airline'])['City1'].size()

low_price_airline_group.plot.bar(figsize=(18,8))

plt.show()
# Printing top 5 rows

df.head()
X = df.drop('Average_Fare0' , axis = 1)

y = df.Average_Fare0
X = pd.get_dummies(X)

X.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV





from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error , make_scorer
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state = 42)
#printing length

len(y_train),len(X_train)
from sklearn.preprocessing import StandardScaler
# creating standardscaler object

scaler = StandardScaler()



#Scaling and Transforming our training Dataframe

X_train = scaler.fit_transform(X_train)

# We don't want our test set to learn from training Data so, we are will just transform it

X_test = scaler.transform(X_test)
kfolds = KFold(n_splits=3, shuffle=True, random_state=42)



# model scoring and validation function

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="r2",cv=kfolds))

    return (rmse)



# rmsle scoring function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
linear_model_rfe = LinearRegression()

linear_model_rfe.fit(X_train,y_train)
## Train Accuracy

y_train_pred=linear_model_rfe.predict(X_train)



## Test Accuracy

y_pred=linear_model_rfe.predict(X_test)



rmsle(y_test,y_pred), rmsle(y_train,y_train_pred)
#### Let's verify this by evaluating r-squared score



from sklearn.metrics import r2_score

r2_score(y_train,y_train_pred)
y_pred = linear_model_rfe.predict(X_test)

r2_score(y_test,y_pred)
### Also Cross Validation Score

cv_rmse(linear_model_rfe,X)
linear_train_score = linear_model_rfe.score(X_train,y_train)

linear_test_score = linear_model_rfe.score(X_test, y_test)

linear_train_score , linear_test_score
from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Selection top 20 features using RFE

select = RFE(linear_model_rfe, 20 ,step=1)

select = select.fit(X_train,y_train)
# Ranking features based on their relevancy

select.ranking_
# Zipping column names, ranking and support

list(zip(X.columns,select.support_,select.ranking_))
col = X.columns[select.support_]

col
X_train_rfe = X[col]
# Checking VIF of Each predictor Variable

vif = pd.DataFrame()

vif['Features']=X_train_rfe.columns

vif['VIF'] = [ variance_inflation_factor(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]

vif.sort_values(by = 'VIF' , ascending=False)
# training and test data with top 20 features given by VIF

xx = pd.DataFrame(X_train, columns= X.columns)[col]

xt = pd.DataFrame(X_test, columns= X.columns)[col]
lm = LinearRegression()

lm.fit(xx,y_train)
## Train Accuracy

y_train_pred=lm.predict(xx)



## Test Accuracy

y_pred=lm.predict(xt)



rmsle(y_test,y_pred), rmsle(y_train,y_train_pred)
from sklearn.metrics import r2_score

r2_score(y_train,y_train_pred)
y_pred = lm.predict(xt)

r2_score(y_test,y_pred)
### Also Cross Validation Score

cv_rmse(lm,X)
linear_train_score = lm.score(xx,y_train)

linear_test_score = lm.score(xt, y_test)

linear_train_score , linear_test_score
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



lasso = Lasso(alpha=0.01, max_iter=10e5)

rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

Ridge_train_score = rr.score(X_train,y_train)

Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score,Ridge_test_score
lasso.fit(X_train, y_train)

Lasso_train_score = lasso.score(X_train,y_train)

Lasso_test_score = lasso.score(X_test, y_test)

Lasso_train_score,Lasso_test_score
plt.figure(figsize = (16,10))

plt.plot(rr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Ridge Regression')

plt.plot(lasso.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='red',label='Lasso Regression')



plt.xlabel('Coefficient Index',fontsize=16)

plt.ylabel('Coefficient Magnitude',fontsize=16)

plt.legend(fontsize=13,loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
coeff_df = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])  

coeff_df['absolute'] = coeff_df.Coefficient.abs()

coeff_df.head()
coeff_df = coeff_df.sort_values(by = 'absolute', ascending = False)



# Printing top 10 variables with highest importance/impact on label per unit change

coeff_df.Coefficient.head(10)