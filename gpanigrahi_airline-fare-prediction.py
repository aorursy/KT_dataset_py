import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import requests

import seaborn as sns

import scipy.stats as ss

import sklearn.linear_model as lm

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('/kaggle/input/airline/airline.csv')
df.columns
df = df.drop('Unnamed: 0',axis=1)
df.describe()
df.info()
df.select_dtypes(include=['object']).nunique()
#No null values in the dataset

df.isnull().values.any()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Market Share')

sns.distplot(df['market share'])



plt.subplot(1,2,2)

plt.title('Market Share.1')

sns.boxplot(y=df['market share'])



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Market Share')

sns.distplot(df['market share.1'])



plt.subplot(1,2,2)

plt.title('Market Share.1')

sns.boxplot(y=df['market share.1'])



plt.show()
print(df['market share.1'].describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Average fare_1')

sns.distplot(df['Average Fare'])



plt.subplot(1,2,2)

plt.title('Average fare_1')

sns.boxplot(y=df['Average Fare'])



plt.show()
#Average fare of

plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Average fare')

sns.distplot(df['Average fare'])



plt.subplot(1,2,2)

plt.title('Average fare_2')

sns.boxplot(y=df['Average fare'])



plt.show()
plt.figure(figsize=(50, 20))



plt.subplot(1,2,2)

plt1 = df['City1'].value_counts().plot('bar')

plt.title('City1 Histogram')

plt1.set(xlabel = 'City1', ylabel='Frequency of City1')



plt.show()
plt.subplot(1,1,1)

plt3 = df['market leading airline'].value_counts().plot('bar')

plt.title('market leading airline Histogram')

plt3.set(xlabel = 'market leading airline', ylabel='Frequency of market leading airline')
plt.subplot(1,1,1)

plt4 = df['Low price airline'].value_counts().plot('bar')

plt.title('Low price airline Histogram')

plt4.set(xlabel = 'Low price airline', ylabel='Frequency of Low price airline')
from IPython.display import display



with pd.option_context('precision', 2):

    display(df.groupby(['City1'])['Average Fare'].describe()[['count', 'mean']])
cats = ['City1','City2','market leading airline','Low price airline']
city = df['City1'].append(df['City2'])

airlines = df['market leading airline'].append(df['Low price airline'])
print('unique locations: {} | unique airlines: {}'.format(city.nunique(), airlines.nunique()))
df[cats] = df[cats].astype('category')
df_1 = df.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)
sns.pairplot(df);
fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(df_1.corr(),vmin=-0.8, annot=True, cmap='coolwarm',ax=ax);
def scatter(x,fig):

    plt.subplot(5,2,fig)

    plt.scatter(df_1[x],df_1['price'])

    plt.title(x+' vs Price')

    plt.ylabel('Price')

    plt.xlabel(x)



plt.figure(figsize=(10,20))



scatter('Average Fare', 1)

scatter('Average fare', 2)



plt.tight_layout()
def pp(x,y,z):

    sns.pairplot(df_1, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')

    plt.show()



pp('market leading airline', 'Average fare', 'market share')



pp('Low price airline', 'Average weekly passengers', 'market share.1')
from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(df_1, train_size = 0.7, test_size = 0.3, random_state = 100)
#Dividing data into X and y variables

y_train = df_train.pop('Average Fare')

X_train = df_train
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_new = build_model(X_train_rfe,y_train)
X_train_new = X_train_rfe.drop(["City2"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
X_train_new = X_train_new.drop(["City1"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new = X_train_new.drop(["Average fare"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["Distance"], axis = 1)
X_train_new
X_train_new = build_model(X_train_new,y_train)
X_train_new = X_train_new.drop(["Average weekly passengers"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
#Dividing into X and y

y_test = df_test.pop('Average Fare')

X_test = df_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16) 
print(lm.summary())