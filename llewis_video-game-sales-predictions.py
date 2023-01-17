# Importing required libraries

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
# Listing the files in the input directory

print(os.listdir("../input"))
# Importing the train and test datasets

train_df = pd.read_csv('../input/train.csv', index_col=0) # Setting the id column as the index

test_df = pd.read_csv('../input/test.csv', index_col=0) # Setting the id column as the index



# Printing the first five lines of train_df

train_df.head()
# Printing the first five lines of test_df

test_df.head()
# Checking the info for each column

train_df.info()
# Checking for null values

train_df.isna().sum()
# Dropping columns containing a large number of null values

cleaned_train_df = train_df.drop(['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'], axis=1)
# Median Year_of_Release

cleaned_train_df.Year_of_Release.median()
# Replacing missing values for Year_of_Release with the median value

cleaned_train_df.Year_of_Release.fillna(cleaned_train_df.Year_of_Release.median(), inplace=True)
# Value counts for Genre

cleaned_train_df.Genre.value_counts()
# Value counts for Publisher

cleaned_train_df.Publisher.value_counts()
# Dropping rows with null values for Genre or Publisher

cleaned_train_df.dropna(subset=['Genre', 'Publisher'], inplace=True)
# Confirming there are no null values remaining

cleaned_train_df.isna().sum()
# Checking data types

cleaned_train_df.info()
# Converting Year_of_Release to an integer

cleaned_train_df.Year_of_Release = cleaned_train_df.Year_of_Release.astype('int64')
# Set the style of the visualization

sns.set(style="white")



# Create a covariance matrix

corr = cleaned_train_df.corr()



# Generate a mask the size of our covariance matrix

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize = (7,5))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5});
# Plotting histograms of the continuous variables

cleaned_train_df.hist(figsize=(10,10));
# Descriptive statistics for continuous variables

cleaned_train_df.describe()
# Square root transforming JP_Sales and NA_Sales

cleaned_train_df['JP_Sales_sqrt'] = np.sqrt(cleaned_train_df.JP_Sales)

cleaned_train_df['NA_Sales_sqrt'] = np.sqrt(cleaned_train_df.NA_Sales)
# Plotting histograms of the transformed continuous variables

cleaned_train_df[['JP_Sales_sqrt', 'NA_Sales_sqrt']].hist(figsize=(10,4));
# Dropping the square root transformed columns

cleaned_train_df.drop(['JP_Sales_sqrt', 'NA_Sales_sqrt'], axis=1, inplace=True)
# Replacing 0 with 0.001 for JP_Sales and NA_Sales

cleaned_train_df.JP_Sales.replace({0: 0.001}, inplace=True)

cleaned_train_df.NA_Sales.replace({0: 0.001}, inplace=True)
# Log-transforming JP_Sales and NA_Sales

cleaned_train_df.JP_Sales = np.log(cleaned_train_df.JP_Sales)

cleaned_train_df.NA_Sales = np.log(cleaned_train_df.NA_Sales)
# Plotting histograms of the continuous variables

cleaned_train_df.hist(figsize=(10,8))
# Plot of value counts for each Platform

print("Number of unique values:", cleaned_train_df.Platform.nunique())

cleaned_train_df.Platform.value_counts().plot(kind='bar', figsize=(12,5))

plt.xlabel('Platform')

plt.ylabel('Frequency');
# Plot of value counts for each Genre

print("Number of unique values:", cleaned_train_df.Genre.nunique())

cleaned_train_df.Genre.value_counts().plot(kind='bar', figsize=(6,5))

plt.xlabel('Genre')

plt.ylabel('Frequency');
# Plot of value counts for each Publisher

print("Number of unique values:", cleaned_train_df.Publisher.nunique())

cleaned_train_df.Publisher.value_counts().plot(kind='bar', figsize=(30,5))

plt.xlabel('Publisher')

plt.ylabel('Frequency');
# Dropping the Publisher column

cleaned_train_df.drop('Publisher', axis=1, inplace=True)
# Value counts for Platform

platform_counts = cleaned_train_df.Platform.value_counts()

platform_counts
# Create a boolean column for whether a category is in the list of categories that has a count of less than 200

uncommon_platforms = cleaned_train_df.Platform.isin(platform_counts.index[platform_counts<200])

# Replace the Platform value for rows with a Platform in the uncommon_platforms list with 'Other'

cleaned_train_df.loc[uncommon_platforms, 'Platform'] = 'Other'
# Confirming this has worked - there are now 16 categories instead of 31

platform_cats = list(cleaned_train_df.Platform.unique())

print(cleaned_train_df.Platform.nunique())

cleaned_train_df.Platform.value_counts()
cleaned_train_df = pd.get_dummies(cleaned_train_df)
# Setting up X (features) and y (target)

X = cleaned_train_df.drop('NA_Sales', axis=1)

y = cleaned_train_df.NA_Sales
# Transforming X and y

# Ran into an error here, so scaling has been removed. Suggested version is something like:

#x_scaler = StandardScaler()

#y_scaler = StandardScaler()

#X = pd.DataFrame(x_scaler.fit_transform(X), columns=X.columns)

#y = pd.Series(y_scaler.fit_transform(y), columns=y.columns)
# Train test split

X_train, X_test, y_train, y_test = train_test_split(X,y)
# This function is copied from: https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm

def stepwise_selection(X, y, 

                       initial_list=[], 

                       threshold_in=0.01, 

                       threshold_out = 0.05, 

                       verbose=True):

    """ Perform a forward-backward feature selection 

    based on p-value from statsmodels.api.OLS

    Arguments:

        X - pandas.DataFrame with candidate features

        y - list-like with the target

        initial_list - list of features to start with (column names of X)

        threshold_in - include a feature if its p-value < threshold_in

        threshold_out - exclude a feature if its p-value > threshold_out

        verbose - whether to print the sequence of inclusions and exclusions

    Returns: list of selected features 

    Always set threshold_in < threshold_out to avoid infinite looping.

    See https://en.wikipedia.org/wiki/Stepwise_regression for the details

    """

    included = list(initial_list)

    while True:

        changed=False

        # forward step

        excluded = list(set(X.columns)-set(included))

        new_pval = pd.Series(index=excluded)

        for new_column in excluded:

            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()

            new_pval[new_column] = model.pvalues[new_column]

        best_pval = new_pval.min()

        if best_pval < threshold_in:

            best_feature = new_pval.idxmin()

            included.append(best_feature)

            changed=True

            if verbose:

                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))



        # backward step

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()

        # use all coefs except intercept

        pvalues = model.pvalues.iloc[1:]

        worst_pval = pvalues.max() # null if pvalues is empty

        if worst_pval > threshold_out:

            changed=True

            worst_feature = pvalues.argmax()

            included.remove(worst_feature)

            if verbose:

                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:

            break

    return included
# Conducting stepwise selection on the training data

final_features = stepwise_selection(X_train, y_train)
len(final_features)
# Final model summary

predictors = sm.add_constant(X_train[final_features])

final_model = sm.OLS(y_train,predictors).fit()

final_model.summary()
final_features
# Fitting a model using the selected features

linreg = LinearRegression()

linreg.fit(X_train[final_features], y_train)



# Predicting y values

y_hat_train = linreg.predict(X_train[final_features])

y_hat_test = linreg.predict(X_test[final_features])



# Calculating MSE

train_mse = mean_squared_error(y_train, y_hat_train)

test_mse = mean_squared_error(y_test, y_hat_test)

print("Train MSE:", train_mse)

print("Test MSE:", test_mse)
# Dropping columns not used in the model

cleaned_test_df = test_df.drop(['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating', 'Publisher'], axis=1)



# Replacing 0 with 0.001 for JP_Sales

cleaned_test_df.JP_Sales.replace({0: 0.001}, inplace=True)



# Log-transforming JP_Sales

cleaned_test_df.JP_Sales = np.log(cleaned_test_df.JP_Sales)
# Combining the same Platform categories together as in the training set

plts = platform_cats

plts.remove('Other')
plts
# Replace the Platform value for rows with a Platform not in the plts list with 'Other'

cleaned_test_df.loc[~cleaned_test_df['Platform'].isin(plts), 'Platform'] = 'Other'
cleaned_test_df.Platform.unique()
# One-hot encoding categorical values

cleaned_test_df = pd.get_dummies(cleaned_test_df)
# Only keeping required columns

cleaned_test_df = cleaned_test_df[final_features]
test_data_notnull = cleaned_test_df[cleaned_test_df.Year_of_Release.notnull()]
test_data_null = cleaned_test_df[cleaned_test_df.Year_of_Release.isna()]
# Predicting y values and adding them as a new column

test_data_notnull['Prediction'] = linreg.predict(test_data_notnull)
# Creating a series with the predictions

predictions = test_data_notnull['Prediction']
# Converting predictions to a dataframe

predictions = pd.DataFrame(predictions)
test_data_null['Prediction'] = cleaned_train_df.NA_Sales.median()
test_data_null.head()
null_predictions = test_data_null['Prediction']
null_predictions = pd.DataFrame(null_predictions)
predictions = predictions.append(null_predictions)
predictions.Prediction = np.exp(predictions.Prediction)
predictions.head()
predictions['Id'] = predictions.index
predictions = predictions[['Id', 'Prediction']]
predictions.head()
# Exporting the final predictions as a .csv

predictions.to_csv('submission.csv', index=False)