# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from random import randrange, uniform

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Day - Dataset.csv",index_col = 0)
df.head(10)
df.columns
df.dtypes
#Converting variables datatype to required datatypes

#Categorical variables

df['dteday'] = pd.to_datetime(df['dteday'],yearfirst = True)

df['season'] = df['season'].astype('category')

df['yr']     = df['yr'].astype('category')

df['mnth']   = df['mnth'].astype('category')

df['holiday']= df['holiday'].astype('category')

df['weekday']= df['weekday'].astype('category')

df['workingday']= df['workingday'].astype('category')

df['weathersit']= df['weathersit'].astype('category')



#Continuous variables

df['temp'] = df['temp'].astype('float')

df['atemp']= df['atemp'].astype('float')

df['hum']  = df['hum'].astype('float')

df['windspeed'] = df['windspeed'].astype('float')

df['casual'] = df['casual'].astype('float')

df['registered'] = df['registered'].astype('float')

df['cnt'] = df['cnt'].astype('float')
df.dtypes
ordered_data = df.copy()
missing_val = pd.DataFrame(df.isnull().sum())

missing_val = missing_val.reset_index()

missing_val = missing_val.rename(columns={'index':'variable',0:'Missing_values'})
missing_val
#Craeting new variables from existing variables for visualizations (Future Engineering)

df['actual_temp'] = df['temp'] * 39

df['actual_atemp'] = df['atemp'] * 50

df['actual_windspeed'] = df['windspeed'] * 67

df['actual_hum'] = df['hum'] * 100
df.columns
df.head()
#Cheking the Distribution of data by using Histograms

sns.set_style("whitegrid")

sns.distplot(df['actual_temp'],rug=True)
sns.distplot(df['actual_atemp'], rug=True)
sns.distplot(df['actual_hum'], rug=True)
sns.distplot(df['actual_windspeed'],rug=True)
sns.distplot(df['cnt'],rug=True)
continuous_variables = ['actual_temp','actual_atemp','actual_windspeed','actual_hum','cnt']

for i in continuous_variables:

    plt.hist(df[i],bins='auto')

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Density")

    plt.xlabel(i)

    plt.show()
#Bike Rentals Per Monthly

monthly_sales = df.groupby('mnth').size()

print(monthly_sales)

#Plotting the Graph

plot = monthly_sales.plot(title='Monthly Sales',xticks=(1,2,3,4,5,6,7,8,9,10,11,12))

plot.set_xlabel('Months')

plot.set_ylabel('Total Number of Bikes')
#Checking the distribution categorical Data using factorplot

sns.set_style("whitegrid")

sns.factorplot(data=df, x='dteday', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='yr', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='mnth', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='season', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='weathersit', kind= 'count',size=4,aspect=2)

sns.factorplot(data=df, x='workingday', kind= 'count',size=4,aspect=2)

#Scatter plot for temprature against bike rentals 

sns.scatterplot(data=df,x='actual_temp',y='cnt')

#Scatter plot for humidity against bike rentals 

sns.scatterplot(data=df,x='actual_hum',y='cnt')

#Scatter plot for atemp(feeled_temparature) against bike rentals 

sns.scatterplot(data=df,x='actual_atemp',y='cnt')
#Scatter plot for windspeed against bike rentals 

sns.scatterplot(data=df,x='actual_windspeed',y='cnt')
df.columns
#Checking Outliers in  data using boxplot

sns.boxplot(data=df[['actual_temp','actual_atemp','actual_windspeed','actual_hum']])

fig=plt.gcf()

fig.set_size_inches(8,8)
sns.boxplot(data=df[['season','mnth','holiday','weekday']])

fig=plt.gcf()

fig.set_size_inches(8,8)
sns.boxplot(data=df[['workingday','weathersit','casual','registered']])

fig=plt.gcf()

fig.set_size_inches(8,8)
#Variables that are used to remove outliers

#Not considering casual because this is not predictor variable

#Not considering holiday because workingday variable includes holiday, so therte is no useful of considering holiday variables.

out_names = ['actual_windspeed','actual_hum']
#Detecting and Removing Outliers

for i in out_names :

    print (i)

    q75,q25 = np.percentile(df.loc[:,i],[75,25])

    iqr = q75-q25

    

    min = q25 - (iqr*1.5)

    max = q75 + (iqr*1.5)

    print (min)

    print (max)

    

    df = df.drop(df[df.loc[:,i] < min].index)

    df = df.drop(df[df.loc[:,i] > max].index)   
#Checking Outliers in data after outliers removel using boxplot

sns.boxplot(data=df[['actual_temp','actual_atemp','actual_windspeed','actual_hum']])

fig=plt.gcf()

fig.set_size_inches(8,8)
df.head()
continuous_variables = [ 'temp','atemp', 'hum', 'windspeed', 'casual',

                        'registered', 'cnt', 'actual_temp', 'actual_atemp', 'actual_windspeed', 'actual_hum']
#Future selection on the basis of Correlation, multcollinearity and variable importance

#cnames = ["actual_temp","actual_atemp","actual_hum","acttual_windspeed"]

#cnames = ["temp","atemp","hum","windspeed"]



df_cor = df.loc[:,continuous_variables]

f, ax = plt.subplots(figsize=(10,10))



#Generate correlation matrix

cor_mat = df_cor.corr()



#Plot using seaborn library

sns.heatmap(cor_mat, mask=np.zeros_like(cor_mat, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.plot()
cat_columns = ['season', 'yr', 'mnth', 'holiday', 'weekday','workingday', 'weathersit']

# making every combinationfrom cat_columns

factors_paired = [(i,j) for i in cat_columns for j in cat_columns]

factors_paired

p_values = [] 

from scipy.stats import chi2_contingency 

for factor in factors_paired:

    if factor[0] != factor[1]:

        chi2, p, dof, ex = chi2_contingency(pd.crosstab(df[factor[0]], df[factor[1]]))

        p_values.append(p.round(3))

    else:

        p_values.append('-') 

p_values = np.array(p_values).reshape((7,7))

p_values = pd.DataFrame(p_values, index=cat_columns, columns=cat_columns)

print(p_values)
# checking vif of numerical column without dropping multicollinear column 

from statsmodels.stats.outliers_influence import variance_inflation_factor as vf 

from statsmodels.tools.tools import add_constant

continuous = add_constant(df[['temp', 'atemp', 'hum', 'windspeed']])

vif = pd.Series([vf(continuous.values, i) for i in range(continuous.shape[1])], index = continuous.columns) 

print(vif.round(1))



# Checking VIF values of numeric columns after dropping column atemp 

from statsmodels.stats.outliers_influence import variance_inflation_factor as vf 

from statsmodels.tools.tools import add_constant

continuous = add_constant(df[['temp', 'hum', 'windspeed']]) 

vif = pd.Series([vf(continuous.values, i)  for i in range(continuous.shape[1])], index = continuous.columns) 

vif.round(1)
#Removing variables atemp beacuse it is highly correlated with temp,

#Removing weekday,holiday because they don't contribute much to the independent cariable

#Removing Causal and registered becuase that's what we need to predict.

df = df.drop(columns=['holiday','dteday','atemp','casual','registered','actual_temp','actual_atemp',

                      'actual_windspeed','actual_hum'])
df.head(10)
df.columns
df2 = df.copy()
categorical_var = ['season', 'yr', 'mnth', 'weekday', 'workingday', 'weathersit']
df2.columns
#Dummy Variable creation for categorical variables

df2 = pd.get_dummies(data = df2,columns=categorical_var)
df2.columns
df2['count'] = df2['cnt'] 

df2 =  df2.drop('cnt',axis=1)

df2.columns
df_plt_tree = df2.drop('count',axis=1) 

df2.shape
#Import Libraries for decision tree

from sklearn.tree import DecisionTreeRegressor,export_graphviz

from sklearn.metrics import accuracy_score,r2_score,mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn import tree
#Splitting data into train and test data

train,test = train_test_split(df2,test_size = 0.2, random_state = 123)
#Function for Performing all the tasks such as Error metrix rmse,mape,r-squared,accuracy,predictions

def evaluate(model, test_features, test_actual):

    predictions = model.predict(test_features)

    #Creating new data frame with comparing actual and predicted values

    df_Dt = pd.DataFrame({'actual':test_actual,'predicted':predictions})

    errors = abs(predictions - test_actual)

    mape = 100 * np.mean(errors / test_actual)

    accuracy = 100 - mape

    rmse = np.sqrt(mean_squared_error(test_actual,predictions))

    rsquared = r2_score(test_actual, predictions)

    print('<---Model Performance--->')

    print('R-Squared Value = {:0.2f}'.format(rsquared))

    print('RMSE = {:0.2f}'.format(rmse))

    print('MAPE = {:0.2f}'.format(mape))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return
#Decision Tree model development

#Training the model with train data

model = DecisionTreeRegressor(random_state = 123).fit(train.iloc[:,0:33],train.iloc[:,33])



#Function for predictions, Error metrix rmse,mape,r-squared,accuracy

evaluate(model, test.iloc[:,0:33], test.iloc[:,33])



dotfile = open("pt.dot",'w')

df = tree.export_graphviz(model,out_file=dotfile,feature_names = df_plt_tree.columns)
#import libraries for Linear regression

from sklearn.linear_model import LinearRegression



#Create model Linear Regression using LinearRegression

model = LinearRegression().fit(train.iloc[:,0:33],train.iloc[:,33])



#Function for predictions, Error metrix rmse,mape,r-squared,accuracy

evaluate(model, test.iloc[:,0:33], test.iloc[:,33])
#Import the libraries for Random Forest

from sklearn.ensemble import RandomForestRegressor

#Train the model

Rf_model = RandomForestRegressor(n_estimators=500,random_state=123).fit(train.iloc[:,0:33], train.iloc[:,33])



#Function for predictions, Error metrix rmse,mape,r-squared,accuracy

evaluate(Rf_model, test.iloc[:,0:33], test.iloc[:,33])
#Hyperparameter tuning using GridSearchCV

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [12,14,16],

    'max_features': ['auto', 'sqrt'],

    'min_samples_leaf': [2,3],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [900,1000,1200]

}

# Create a based model

rf = RandomForestRegressor()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(test.iloc[:,0:33], test.iloc[:,33])

grid_search.best_params_

best_grid = grid_search.best_estimator_

#Applying gridsearchcsv to test data

grid_accuracy = evaluate(best_grid,test.iloc[:,0:33],test.iloc[:,33])