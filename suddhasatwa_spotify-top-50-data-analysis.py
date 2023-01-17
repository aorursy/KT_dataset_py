# Importing all required libraries. 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



# Disabling in-line warnings in the Notebook. 

warnings.filterwarnings('ignore')
# Loading the Raw data. 

spotify = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='cp1252')
# Looking at the sample rows of the Raw data. 

spotify.head(10)
# Drop the unnamed column. 

spotify.drop('Unnamed: 0', axis=1, inplace=True)
# Let's check the overall summary statistics of the numeric fields in the Dataframe.

spotify.describe()
# Checking sum of NA values. 

spotify.isna().sum()
# Checking the song with Max energy. 

spotify[spotify.Energy == np.max(spotify.Energy)]
# Checking the song with Max Danceability. 

spotify[spotify.Danceability == np.max(spotify.Danceability)]
# Checking the song with Max Loudness. 

spotify[spotify['Loudness..dB..'] == np.max(spotify['Loudness..dB..'])]
# Checking the song with Max Danceability. 

spotify[spotify['Liveness'] == np.max(spotify['Liveness'])]
# Checking the song with Max Danceability. 

spotify[spotify['Length.'] == np.max(spotify['Length.'])]
# Checking the song with Max Danceability. 

spotify[spotify['Popularity'] == np.max(spotify['Popularity'])]
# Checking structure of the dataframe. 

spotify.info()
# Checking Histogram of artist name. 

plt.figure(figsize=(20,10))

sns.countplot(spotify['Artist.Name'])

plt.show()
# Checking Histogram of artist name. 

plt.figure(figsize=(20,10))

sns.countplot(spotify['Genre'])

plt.show()
# Checking all rows of Genre column. 

spotify['Genre']
# Selecting rows where Genre contains the word "pop"

spotify[spotify['Genre'].str.contains('pop')]
# Checking the count of Songs with Pop Genre

spotify[spotify['Genre'].str.contains('pop')].count()
# Checking rows with Genre other than Pop

spotify[~spotify['Genre'].str.contains('pop')]
# Checking the Latin songs. 

spotify[spotify['Genre'].str.contains('latin')].count()
# Checking the Rap songs. 

spotify[spotify['Genre'].str.contains('rap')].count()
# Checking the Hip Hop songs. 

spotify[spotify['Genre'].str.contains('hip')].count()
# Imputing values for Pop Genre

spotify.loc[spotify['Genre'].str.contains('pop', case=False), 'Genre'] = 'Pop'
# Imputing values for Latin Genre

spotify.loc[spotify['Genre'].str.contains('latin', case=False), 'Genre'] = 'Latin'
# Imputing values for Rap Genre

spotify.loc[spotify['Genre'].str.contains('rap', case=False), 'Genre'] = 'Rap'
# Imputing values for Hip Hop Genre

spotify.loc[spotify['Genre'].str.contains('hip', case=False), 'Genre'] = 'Hip-Hop'
# Checking the final status of Genre column. 

spotify.Genre
# Checking distribution of the Genre column. 

plt.figure(figsize=(20,10))

sns.countplot(spotify['Genre'])

plt.show()
# Checking the list of columns in our dataset. 

spotify.columns
# Checking pairplots of all variables first. 

plt.figure(figsize=(20,10))

sns.pairplot(spotify, hue='Genre')

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (20, 10))

sns.heatmap(spotify.corr(), annot = True, cmap="YlGnBu")

plt.show()
# Checking the list of columns in our data. 

spotify.columns
# Dropping the Track Name column. 

spotify.drop('Track.Name', axis=1, inplace=True)
# Checking the columns again.

spotify.columns
# Dropping the Artist Name column. 

spotify.drop('Artist.Name', axis=1, inplace=True)
# importing library for label encoding the Genre Data. 

from sklearn.preprocessing import LabelEncoder
# creating object for label encoding. 

le = LabelEncoder()
# Encoding the Genre column. 

spotify.Genre = le.fit_transform(spotify.Genre)
# Checking the dataset information. 

spotify.info()
# Checking the sample data. 

spotify.head(10)
# Check the distribution of target variable. 

plt.figure(figsize=(20,10))

sns.distplot(spotify.Popularity)

plt.show()
# Creating the Features and Targets datasets. 

X = spotify[['Genre', 'Beats.Per.Minute', 'Energy', 'Danceability',

       'Loudness..dB..', 'Liveness', 'Valence.', 'Length.', 'Acousticness..',

       'Speechiness.']]



y = spotify.Popularity
# Importing library for Train Test split. 

from sklearn.model_selection import train_test_split
# Creating the splits. 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# Checking the Training data. 

X_train
# Checking the dimensions of the training and testing sets. 

print("Training Feature data : ", X_train.shape)

print("Training Feature data : ", X_test.shape)

print("Training Feature data : ", y_train.shape)

print("Testing Target data : ", y_test.shape)
# Importing library for standard scaling

from sklearn.preprocessing import StandardScaler
# Creating the scaler object

scaler = StandardScaler()
# Scaling the Training and Testing Data. 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Checking the dimensions of the training and testing sets. 

print("Training Feature data : ", X_train.shape)

print("Training Feature data : ", X_test.shape)

print("Training Feature data : ", y_train.shape)

print("Testing Target data : ", y_test.shape)
# Checking the training data. 

X_train
# Import the libraries .

from sklearn.linear_model import LinearRegression
# Creating the object

regressor = LinearRegression()
# Fit the model. 

regressor.fit(X_train, y_train)
# Predicting the test results. 

y_pred = regressor.predict(X_test)
# Checking the predictions. 

y_pred
# Checking the actuals

y_test
# Checking the model coefficients. 

regressor.coef_
# spotify dataset columns. 

X.columns
# Creating dataframe of features and coefficients. 

output = {'Features': X.columns, 'Coefficient': regressor.coef_}

output_df = pd.DataFrame(output)

output_df
# Checking RMSE



# Import libraries. 

from sklearn.metrics import mean_squared_error



# Checking the RMSE

mean_squared_error(y_pred, y_test)
# Checking the intercept

regressor.intercept_
# Importing the RFE Library. 

from sklearn.feature_selection import RFE
# Running RFE with the output number of the variable equal to 5

# We select 5 as we have total 10 variables, hence 5 looks to be a good number,

# Considering we do not loose much information from the functional perspective as well .. !! 

rfe = RFE(regressor, 5) # running RFE

rfe = rfe.fit(X_train, y_train) # Fitting the training data
# Getting the columns with RFE

list(zip(X.columns,rfe.support_,rfe.ranking_))
# Getting total list of columns. 

X.columns
# Creating new set of features. 

X_new = X[['Energy', 'Valence.', 'Length.', 'Acousticness..', 'Speechiness.']]
# Creating new test and train sets. 

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)
# Scaling the Training and Testing Data. 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Creating the object

regressor_rfe = LinearRegression()
# Fitting the model. 

regressor_rfe.fit(X_train, y_train)
# Getting the predictions. 

y_pred_rfe = regressor_rfe.predict(X_test)
# Checking the RMSE

mean_squared_error(y_pred_rfe, y_test)
# Checking the intercept

regressor_rfe.intercept_
# Creating dataframe of features and coefficients. 

output_rfe = {'Features': X_new.columns, 'Coefficient': regressor_rfe.coef_}

output_df_rfe = pd.DataFrame(output_rfe)

output_df_rfe
# Creating dataframe of actuals and predictions for side by side comparisons. 

# We will compare the differences as well, if possible. 

prediction_diff = y_pred_rfe - y_test

check_predictions = {'Predictions': y_pred_rfe, 'Actuals': y_test, 'Difference': prediction_diff}

check_predictions_df = pd.DataFrame(check_predictions)

check_predictions_df
# Checking shape of the output predictions comparisons' dataframe. 

check_predictions_df.shape
# Import the required library. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculate the VIFs for all the variables/features in our dataset. 



vif_all = pd.DataFrame()



vif_all['Features'] = X.columns



vif_all['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



vif_all['VIF'] = round(vif_all['VIF'], 2)



vif_all = vif_all.sort_values(by = "VIF", ascending = False)



vif_all
# Calculate the VIFs for the new model



vif = pd.DataFrame()



vif['Features'] = X_new.columns



vif['VIF'] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]



vif['VIF'] = round(vif['VIF'], 2)



vif = vif.sort_values(by = "VIF", ascending = False)



vif
# Dropping the Energy column from our new dataset and checking the VIF again. 



X_new.drop('Energy', axis=1, inplace=True)



vif_2 = pd.DataFrame()



vif_2['Features'] = X_new.columns



vif_2['VIF'] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]



vif_2['VIF'] = round(vif_2['VIF'], 2)



vif_2 = vif_2.sort_values(by = "VIF", ascending = False)



vif_2
# Creating new test and train sets. 

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)
# Scaling the Training and Testing Data. 

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Creating the object

regressor_vif = LinearRegression()
# Fitting the model. 

regressor_vif.fit(X_train, y_train)
# Making predictions. 

y_pred_vif = regressor_vif.predict(X_test)
# Creating dataframe of actuals and predictions for side by side comparisons. 

# We will compare the differences as well, if possible. 

prediction_diff = y_pred_vif - y_test

check_predictions = {'Predictions': y_pred_vif, 'Actuals': y_test, 'Difference': prediction_diff}

check_predictions_df = pd.DataFrame(check_predictions)

check_predictions_df
# Checking the RMSE

mean_squared_error(y_pred_vif, y_test)
# Checking the intercept

regressor_vif.intercept_
# Creating dataframe of features and coefficients. 

output_vif = {'Features': X_new.columns, 'Coefficient': regressor_vif.coef_}

output_df_vif = pd.DataFrame(output_vif)

output_df_vif
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure(figsize=(20,10))

plt.scatter(y_test,y_pred_vif)

fig.suptitle('Actuals v/s Predicted', fontsize=20)              # Plot heading 

plt.xlabel('Actuals', fontsize=18)                          # X-label

plt.ylabel('Predicted', fontsize=16)                          # Y-label
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_test - y_pred_vif), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label