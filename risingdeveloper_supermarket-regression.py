import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Makes graph display in notebook

%matplotlib inline   
supermarket_data = pd.read_csv('../input/train.csv')
supermarket_data.head()
#Displaty the number of rows and columns

supermarket_data.shape
supermarket_data.describe()
supermarket_data.dtypes
#Remove Id columns

cols_2_remove = ['Product_Identifier', 'Supermarket_Identifier', 'Product_Supermarket_Identifier']



new_data = supermarket_data.drop(cols_2_remove, axis=1)
new_data.head()
cat_cols = ['Product_Fat_Content','Product_Type',

            'Supermarket _Size', 'Supermarket_Location_Type',

           'Supermarket_Type' ]



num_cols = ['Product_Weight', 'Product_Shelf_Visibility',

            'Product_Price', 'Supermarket_Opening_Year', 'Product_Supermarket_Sales']
for col in cat_cols:

    print('Value Count for', col)

    print(new_data[col].value_counts())

    print("---------------------------")
counts = new_data['Supermarket_Type'].value_counts() # find the counts for each unique category

counts
colors = ['green', 'red', 'blue', 'yellow', 'purple']



for i,col in enumerate(cat_cols):

    fig = plt.figure(figsize=(6,6)) # define plot area

    ax = fig.gca() # define axis  

    

    counts = new_data[col].value_counts() # find the counts for each unique category

    counts.plot.bar(ax = ax, color = colors[i]) # Use the plot.bar method on the counts data frame

    ax.set_title('Bar plot for ' + col)



new_data.head(3)
for col in num_cols:

    fig = plt.figure(figsize=(6,6)) # define plot area

    ax = fig.gca() # define axis  



    new_data.plot.scatter(x = col, y = 'Product_Supermarket_Sales', ax = ax)

for col in cat_cols:

    sns.set_style("whitegrid")

    sns.boxplot(col, 'Product_Supermarket_Sales', data=new_data)

    plt.xlabel(col) # Set text for the x axis

    plt.ylabel('Product Supermarket Sales')# Set text for y axis

    plt.show()

  
#save the target value

y_target = new_data['Product_Supermarket_Sales']

new_data.drop(['Product_Supermarket_Sales'], axis=1, inplace=True)
new_data.head(2)
# dummy_data = pd.get_dummies(new_data)

# dummy_data.head()
from sklearn.preprocessing import LabelEncoder
for cat in cat_cols:

    lb = LabelEncoder()

    lb.fit(list(new_data[cat].values))

    new_data[cat] = lb.transform(list(new_data[cat].values))
new_data.head()
new_data.isnull().sum()
mean_pw = np.mean(new_data['Product_Weight'])
new_data['Product_Weight'].fillna(mean_pw, inplace=True)
new_data.isnull().sum()
new_data.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(new_data)



scaled_data = scaler.transform(new_data)
# Split our data into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y_target, test_size = 0.3)
print("Shape of train data", X_train.shape)

print("Shape of train target ", y_train.shape)

print("Shape of test data", X_test.shape)

print("Shape of test target", y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor 

import xgboost as xgb

from sklearn.metrics import mean_absolute_error
# Using Linear Model

lm = LinearRegression()

lm.fit(X_train, y_train)



#Prediction

predictions_lm = lm.predict(X_test)



#Calculate error

lm_error = mean_absolute_error(y_test, predictions_lm)

print("Mean Absolute Error for Linear model is", lm_error)
# Using Linear Model

rand_model = RandomForestRegressor(n_estimators=400, max_depth=6)

rand_model.fit(X_train, y_train)



#Prediction

predictions_rf = rand_model.predict(X_test)



#Calculate error

rf_error = mean_absolute_error(y_test, predictions_rf)

print("Mean Absolute Error for Random Forest model is", rf_error)
# Using ensemble technique

xgb_model = xgb.XGBRegressor(max_depth=4, n_estimators=500, learning_rate=0.1)



xgb_model.fit(X_train, y_train)



#Prediction

predictions_xgb = xgb_model.predict(X_test)



#Calculate error

xgb_error = mean_absolute_error(y_test, predictions_xgb)

print("Mean Absolute Error for XGB model is", xgb_error)