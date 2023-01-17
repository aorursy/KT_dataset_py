# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np                 # linear algebra

import pandas as pd                # data processing 

import matplotlib.pyplot as plt    # data visualization

import seaborn as sns              # data visualization
# out of many brand data file we are focussing only on Skoda

df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/skoda.csv')

df.head(15)
# to view various info of available data

df.info()
# the shape of our data 

df.shape
# to look for missing values

df.notnull()
# to calculate vehicle age 

df['age'] = 2020 - df['year']

df = df.drop(columns = 'year')    # removed that column

df.head()
# to count number of zero value in each column

df.isin([0]).sum()
# totally we have these much zero values for 'engineSize' and 'tax'

# the age zero values represents the car purchased year on 2020

print(sum(df['engineSize'] == 0))

print(sum(df['tax'] == 0))
# the zero values 

df[["engineSize","tax"]] = df[["engineSize","tax"]].replace(0,np.NaN)   # replacing Zero by Nan values

df.isnull().sum()
median_to_fill = df.groupby("model").median()       # Groupby.Median: Compute median of groups, excluding missing values. 



for model, row in median_to_fill.iterrows():        # Iterrows: Iterate over DataFrame rows

    rows_to_fill = (df["model"] == model)

    df[rows_to_fill] = df[rows_to_fill].fillna(row) # Fillna: Fills the NaN values with a given substitute number
# to count number of zero value in each column

df.isin([0]).sum()
df.head(15)
df.describe() 
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot=True) # annot=True shows the values inside the box
df.corr().abs()   # abs: generates the absolute value
numeric_type = df[['price','mileage','tax','engineSize','mpg','age']] # Includes only Int and Float type Datas



fig = plt.figure(figsize=(15,10))                             # Figsize: Dimension of the plot 

                                   

for index,col in enumerate(numeric_type):                     # Enumerate iterates over the numeric_type

    sns.set_style('whitegrid')                                # Background theme of the plot

    plt.subplot(3,3,index+1)                                  # Controls the rows and columns, Index for placing

    sns.set(font_scale = 1.0)                                 # Scaling the Font size

    sns.distplot(df[col],kde = False, color='blue')           # Dist plot: univariate distribution of observations

fig.tight_layout(pad=1.0)                                     # To adjust subplots Label 
category_type = df[['model','transmission','fuelType']]         # Includes only Object type Datas



fig = plt.figure(figsize=(20,5))



for index,col in enumerate(category_type):

    sns.set_style('whitegrid')

    plt.subplot(1,3,index+1)

    if(index == 0):

        plt.xticks(rotation=90)                                   # To make X-axis Label in vertical represantation

    sns.set(font_scale = 1.0)

    sns.countplot(df[col], order = df[col].value_counts().index)  # categorical bin using bars representation

fig.tight_layout(pad=1.0)  
# various numeric_type data influence over price

numeric_type = df[['mileage','tax','engineSize','mpg','age']]               # Updated the set without Price



fig = plt.figure(figsize=(20,20))



for index,col in enumerate(numeric_type):                                   

        sns.set_style('whitegrid')                                          

        plt.subplot(4,3,index+1)                   

        sns.set(font_scale = 1.0)

        sns.scatterplot(data = df, x = col, y = 'price',color='blue', alpha = 0.5) 

fig.tight_layout(pad=1.0)   
# various category_type data influence over price



fig = plt.figure(figsize=(20,5))



for index,col in enumerate(category_type):

    sns.set_style('whitegrid')

    plt.subplot(1,3,index+1)

    if(index == 0):

        plt.xticks(rotation=90)

    sns.set(font_scale = 1.0)

    sns.barplot(x=df[col], y='price', data = df, ci = None) # ci: to avoid error bars

fig.tight_layout(pad=1.0)
# Converting all the categorial data into some useful numerical data for better evaluation using One-hot Encoding

from sklearn.preprocessing import OneHotEncoder                                     # To perform encoding of data

df_onehot = pd.get_dummies(df,columns=['model', 'transmission','fuelType'])         # Encoding shown columns 

print(df_onehot.shape)

df_onehot.head()
#Splitting the Train and Test data

from sklearn.model_selection import train_test_split         # Splitting up the data as Train and Test set respectively

X = df_onehot.drop(columns=['price'])                        # X includes all data except target variable

y = df_onehot['price'].copy()                                # y has only target variable-Price

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size = 0.30) # Test size 30%
y_train.shape
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score                         # Evaluate a score by cross-validation

from sklearn.metrics import r2_score                                        # Coefficient of determination 



# Finding the best fit algorithm for our model

model_list = [(LinearRegression(), 'LinearRegression'),                     # List included all desired algorithms

              (GradientBoostingRegressor(),'GradientBoostingRegressor'),

              (DecisionTreeRegressor(),'DecisionTreeRegressor'),

              (RandomForestRegressor(),'RandomForestRegressor'),

              ]



model_score = []



for i in model_list:

    model = i[0]                                                           # Scoring: Coefficient of determination r2

    score = cross_val_score(model,X_train,y_train,cv=4, scoring='r2')      # model: estimator, cv: splitting strategy

    print(f'{i[1]} score = {score.mean().round(2)*100}')                   # Score.mean: Shows mean of all scores                                     

    model_score.append([i[1],score.mean()])
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

# selecting the hyperparameter

param_grid = [

    {'n_estimators' : [200,300,500],                 # number of boosting stages to perform

          'max_depth' : [2,4,6],                     # maximum depth limits the number of nodes in the tree 

          'learning_rate' : [0.1,0.3,0.5]}           # learning rate: Shrinks the contributiuon of each tree                          

]



# Through grid search finding the best Model

grid_search = GridSearchCV(GradientBoostingRegressor(), # estimator object

                           param_grid,                  # includes hyperparameter values

                           cv=4,                        # cross-validation splitting strategy     

                           scoring = 'r2')              # coefficient of determination
grid_search.fit(X_train,y_train)                   # fitting the values in to grid search 

y_pred = grid_search.predict(X_test)               # predicting the Price value 

my_model = grid_search.best_estimator_             # Best estimator has the parameters of better perfomance

my_model                                           # Best model 
my_model.fit(X_train,y_train)                     # Training the best model with datas

prediction = my_model.predict(X_test)             # Predicting the Price values
# To generate a comparison table between predicted and actual Price of Car

result = X_test.copy()

result["predicted"] = my_model.predict(X_test)

result["actual"]= y_test.copy()

result =result[['predicted', 'actual']]

result['predicted'] = result['predicted'].round(2)

result.sample(10)
# Data visulaization of actual price and predicted price of Car

XX = np.linspace(0, 40000, 1881)                                 # return numbers in selected range 

plt.scatter(XX, y_pred, color="green", alpha = 0.2)              # green dots represents y_pred against XX         

plt.scatter(XX, y_test, color="blue", alpha = 0.5)               # blue dots represents y_test against XX
import warnings

warnings.simplefilter("ignore")
# Result visualization

from yellowbrick.regressor import PredictionError   # To plot prediction error

# Instantiate the linear model and visualizer

visualizer = PredictionError(my_model)

visualizer.fit(X_train, y_train)                    # Fit the training data to the visualizer

visualizer.score(X_test, y_test)                    # Evaluate the model on the test data

visualizer.show()                                   # Finalize and render the figure