# Import Libraries

import pandas as pd

import numpy as np

import re

import math



# Visualisations

import matplotlib.pyplot as plt

import seaborn as sns



# Stats and Metrics

from sklearn import metrics

from scipy import stats



# Models

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import GradientBoostingRegressor



# File

df = pd.read_csv('../input/Melbourne_housing_FULL.csv')

df.info()
# Let's assess variables

df.info()
# Covert objects to categorical variables

change_objects = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname']

for colname in change_objects:

    df[colname] = df[colname].astype('category')  

    

# Convert numerical variable to categorical

change_numeric = ['Postcode']

for colname in change_numeric:

    df[colname] = df[colname].astype('category')

    

# Convert date to object  

df['Date'] = pd.to_datetime(df['Date'])



# Check it worked

df.info()
# Compare Rooms and Bedroom2 variables

df ['Rooms v Bedroom2'] = df['Rooms'] - df['Bedroom2']

df.head(100)
# Drop Bedroom2 and Rooms v Bedroom2

df = df.drop(['Bedroom2', 'Rooms v Bedroom2'], 1)
# Check min, max and mean of values to ensure it makes sense

df.describe().transpose()
# Remove false BuildingArea

df = df[df['BuildingArea']!=0]



# Remove false YearBuilt (Melbourne Founded 1835)

df = df[df['YearBuilt']> 1835]
# Display total number of null values

df.isnull().sum()
# Showed that dropping rows is better

df.dropna(inplace = True)



# Uncomment below to initiate either mean or median imputation

# not_null = ['Price', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Lattitude', 'Longtitude']



# Uncomment for Null Values to Mean (ensure dropping rows above and median below are both commented)

#for colname in not_null:

    #df[colname].fillna(df[colname].mean(), inplace = True)



# Uncomment for Null Values to Median (ensure dropping rows and mean above are both commented)

#for colname in not_null:

    #df[colname].fillna(df[colname].median(), inplace = True)
# Build Histogram to visualise price distribution

num_bins = 50

n, bins, patches = plt.hist(df.Price, num_bins, color='b', alpha=0.5, histtype = 'bar', ec = 'black')

plt.ylabel ('Frequency')

plt.xlabel ('Price ($)')

plt.xlim([0, 6000000])

plt.title ('Histogram House Prices')

plt.show()
# Determine Numerical Values

df.select_dtypes(['float64', 'int64']).columns



# Pairplot variables to visualise inter-variable relationships

pair_plot = sns.pairplot(df[['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize','BuildingArea', 'Propertycount', 'YearBuilt', 'Type']], hue = 'Type')
# Build Heatmap to visualise correlations

fig, ax = plt.subplots(figsize=(15,15)) 

heat_map = sns.heatmap(df[df["Type"] == "h"].corr(), cmap = 'jet', annot=True)

# Check in on dataframe

df.info()
# Create features (x) and target (y)

X = df[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']]

y = df['Price']



# Apply MinMaxScaler

scaler = MinMaxScaler()

x = scaler.fit_transform(X)
# Uncomment to use PCA

#pca = PCA()

#pca.fit(x)

#x = pca.transform(x)
# Split the training data and test data

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)
# Initiate max R^2 score

max_r2 = 0



# Create Gradient Boosting Regression model that iterates through learning rates 

for i in np.linspace(0.1, 1, 50):

    

    # Initiate model for learning rate i

    gbr = GradientBoostingRegressor(learning_rate = i)

    gbr.fit(x_train, y_train)

    

    # Make prediction

    y_pred = gbr.predict(x_test)

    

    # Return values for corresponding learning rate

    print ('For learning rate i: %0.2f' %i)

    print('Gradient Boosting Regression MAE: %0.5f'%metrics.mean_absolute_error(y_test,y_pred))

    print('Gradient Boosting MSE:%0.5f'%metrics.mean_squared_error(y_test,y_pred))

    print('Gradient Boosting RMSE:%0.5f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

    print('Gradient Boosting R^2: %0.5f' %metrics.explained_variance_score(y_test,y_pred))

    print ('---------------------------------')



    # If R^2 new maximum score, save score and the learning rate

    if metrics.explained_variance_score(y_test,y_pred) > max_r2:

        max_r2 = metrics.explained_variance_score(y_test,y_pred)

        max_i = i

        y_pred_gbr = y_pred

        

        # Store Standard Error

        se_gbr = stats.sem(y_pred_gbr)



# Print maximum R^2 score and corresponding learning rate

print ('Max R^2 is: %0.5f' %max_r2, 'with learning rate: %0.2f' %max_i)

# Plot residual Plot of the GBR

plt.scatter(y_test, y_pred_gbr, c = 'blue')

plt.ylim([200000, 1000000])

plt.xlim([200000, 1000000])

plt.xlabel("Prices")

plt.ylabel("Predicted prices:")

plt.title("GBR Residual Plot")
# Initialise Linear Regression model

lr = LinearRegression()

lr.fit(x_train, y_train)



# Make Prediction

y_pred_lr = lr.predict(x_test)



# Return Results

print('Linear Regression MAE: %0.5f'%metrics.mean_absolute_error(y_test,y_pred))

print('Linear Regression MSE:%0.5f'%metrics.mean_squared_error(y_test,y_pred))

print('Linear Regression RMSE:%0.5f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('Linear Regression R^2: %0.5f' %metrics.explained_variance_score(y_test,y_pred))



# Store Standard Error

se_lr = stats.sem(y_pred_lr)
# Plot residual Plot of the LR

plt.scatter(y_test, y_pred_lr, c = 'black')

plt.ylim([200000, 1000000])

plt.xlim([200000, 1000000])

plt.xlabel("Prices")

plt.ylabel("Predicted prices:")

plt.title("LR Residual Plot")
# Initialise Lasso Regression model

lcv = LassoCV()

lcv.fit(x_train, y_train)



# Make Prediction

y_pred_lcv = lcv.predict(x_test)



# Return Results

print('Lasso Regression MAE: %0.5f'%metrics.mean_absolute_error(y_test,y_pred))

print('Lasso Regression MSE:%0.5f'%metrics.mean_squared_error(y_test,y_pred))

print('Lasso Regression RMSE:%0.5f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('Lasso Regression R^2: %0.5f' %metrics.explained_variance_score(y_test,y_pred))



se_lcv = stats.sem(y_pred_lcv)
# Plot residual Plot of the LCV

plt.scatter(y_test, y_pred_lcv, c = 'yellow')

plt.ylim([200000, 1000000])

plt.xlim([200000, 1000000])

plt.xlabel("Prices")

plt.ylabel("Predicted prices:")

plt.title("LCV Residual Plot")
# Initialise Max R^2 variable 

max_r2 = 0



# Create Random Forest Model that iterates between 64 --> 128 trees

for n_trees in range (64, 129):

    

    # Initiate model for value n_tree

    rfr = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1) 

    rfr.fit(x_train, y_train)

    

    # Make prediction for n_tree sized model

    y_pred = rfr.predict(x_test)

    

    # Store Standard Error

    rfr_sem = stats.sem (y_pred)

    

    # Print Results

    print('For a Random Forest with', n_trees, 'trees in total:')

    print('MAE: %0.5f'%metrics.mean_absolute_error(y_test,y_pred))

    print('MSE:%0.5f'%metrics.mean_squared_error(y_test,y_pred))

    print('RMSE:%0.5f'%np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

    print('R^2: %0.5f' %metrics.explained_variance_score(y_test,y_pred))

    print('--------------------------------------')

    

    # If new R^2 the max, store it for reference

    if metrics.explained_variance_score(y_test,y_pred) > max_r2:

        max_r2 = metrics.explained_variance_score(y_test,y_pred)

        max_n_trees = n_trees

        max_rfr_sem = rfr_sem

        y_pred_rfr= y_pred

        

        # Store Standard Error

        se_rfr = stats.sem(y_pred_rfr)



# Return max R^2 and corresponding amount of trees in forest

print ('Max R^2 is: %0.5f' %max_r2, 'at', max_n_trees, 'trees')
# Plot residual Plot of the LR

plt.scatter(y_test, y_pred_rfr, c = 'green')

plt.ylim([200000, 1000000])

plt.xlim([200000, 1000000])

plt.xlabel("Prices")

plt.ylabel("Predicted prices:")

plt.title("RFR Residual Plot")