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
# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None
# Display up to 60 columns of a dataframe

pd.set_option('display.max_columns', 60)
# Matplotlib visualization

import matplotlib.pyplot as plt
# Set default font size

plt.rcParams['font.size'] = 24
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize
# Seaborn for visualization

import seaborn as sns

sns.set(font_scale = 2)
# Read in data into a dataframe 

data = pd.read_csv('../input/buildingenergydata/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')



# Display top of dataframe

data.head()
# See the column data types and non-missing values

data.info()
# Replace all occurrences of Not Available with numpy not a number

data = data.replace({'Not Available': np.nan})



# Iterate through the columns

for col in list(data.columns):

    # Select columns that should be numeric

    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in 

        col or 'therms' in col or 'gal' in col or 'Score' in col):

        # Convert the data type to float

        data[col] = data[col].astype(float)
# Statistics for each column

data.describe()
# Function to calculate missing values by column

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(data)
# Get the columns with > 50% missing

missing_df = missing_values_table(data);

missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)

print('We will remove %d columns.' % len(missing_columns))
# Drop the columns

data = data.drop(columns = list(missing_columns))



# For older versions of pandas (https://github.com/pandas-dev/pandas/issues/19078)

# data = data.drop(list(missing_columns), axis = 1)
missing_values_table(data)
figsize(8, 8)



# Rename the score 

data = data.rename(columns = {'ENERGY STAR Score': 'score'})



# Histogram of the Energy Star Score

plt.style.use('fivethirtyeight')

plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');

plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 

plt.title('Energy Star Score Distribution');
# Histogram Plot of Site EUI

figsize(8, 8)

plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');

plt.xlabel('Site EUI'); 

plt.ylabel('Count'); plt.title('Site EUI Distribution');
data['Site EUI (kBtu/ft²)'].describe()
data['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10)
data.loc[data['Site EUI (kBtu/ft²)'] == 869265, :]
# Calculate first and third quartile

first_quartile = data['Site EUI (kBtu/ft²)'].describe()['25%']

third_quartile = data['Site EUI (kBtu/ft²)'].describe()['75%']



# Interquartile range

iqr = third_quartile - first_quartile



# Remove outliers

data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &

            (data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]
# Histogram Plot of Site EUI

figsize(8, 8)

plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');

plt.xlabel('Site EUI'); 

plt.ylabel('Count'); plt.title('Site EUI Distribution');
figsize(8, 8)



# Rename the score 

data = data.rename(columns = {'ENERGY STAR Score': 'score'})



# Histogram of the Energy Star Score

plt.style.use('fivethirtyeight')

plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');

plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 

plt.title('Energy Star Score Distribution');
# Create a list of buildings with more than 100 measurements

types = data.dropna(subset=['score'])

types = types['Largest Property Use Type'].value_counts()

types = list(types[types.values > 100].index)
# Plot of distribution of scores for building categories

figsize(12, 10)



# Plot each building

for b_type in types:

    # Select the building type

    subset = data[data['Largest Property Use Type'] == b_type]

    

    # Density plot of Energy Star scores

    sns.kdeplot(subset['score'].dropna(),

               label = b_type, shade = False, alpha = 0.8);

    

# label the plot

plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 

plt.title('Density Plot of Energy Star Scores by Building Type', size = 28);
# Create a list of boroughs with more than 100 observations

boroughs = data.dropna(subset=['score'])

boroughs = boroughs['Borough'].value_counts()

boroughs = list(boroughs[boroughs.values > 100].index)
# Plot of distribution of scores for boroughs

figsize(12, 10)



# Plot each borough distribution of scores

for borough in boroughs:

    # Select the building type

    subset = data[data['Borough'] == borough]

    

    # Density plot of Energy Star scores

    sns.kdeplot(subset['score'].dropna(),

               label = borough);

    

# label the plot

plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20); 

plt.title('Density Plot of Energy Star Scores by Borough', size = 28);
# Find all correlations and sort 

correlations_data = data.corr()['score'].sort_values()



# Print the most negative correlations

print(correlations_data.head(15), '\n')



# Print the most positive correlations

print(correlations_data.tail(15))
# Select the numeric columns

numeric_subset = data.select_dtypes('number')



# Create columns with square root and log of numeric columns

for col in numeric_subset.columns:

    # Skip the Energy Star Score column

    if col == 'score':

        next

    else:

        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])

        numeric_subset['log_' + col] = np.log(numeric_subset[col])



# Select the categorical columns

categorical_subset = data[['Borough', 'Largest Property Use Type']]



# One hot encode

categorical_subset = pd.get_dummies(categorical_subset)



# Join the two dataframes using concat

# Make sure to use axis = 1 to perform a column bind

features = pd.concat([numeric_subset, categorical_subset], axis = 1)



# Drop buildings without an energy star score

features = features.dropna(subset = ['score'])



# Find correlations with the score 

correlations = features.corr()['score'].dropna().sort_values()
# Display most negative correlations

correlations.head(15)
# Display most positive correlations

correlations.tail(15)
figsize(12, 10)



# Extract the building types

features['Largest Property Use Type'] = data.dropna(subset = ['score'])['Largest Property Use Type']



# Limit to building types with more than 100 observations (from previous code)

features = features[features['Largest Property Use Type'].isin(types)]



# Use seaborn to plot a scatterplot of Score vs Log Source EUI

sns.lmplot('Site EUI (kBtu/ft²)', 'score', 

          hue = 'Largest Property Use Type', data = features,

          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,

          size = 12, aspect = 1.2);



# Plot labeling

plt.xlabel("Site EUI", size = 28)

plt.ylabel('Energy Star Score', size = 28)

plt.title('Energy Star Score vs Site EUI', size = 36);
# Extract the columns to  plot

plot_data = features[['score', 'Site EUI (kBtu/ft²)', 

                      'Weather Normalized Source EUI (kBtu/ft²)', 

                      'log_Total GHG Emissions (Metric Tons CO2e)']]



# Replace the inf with nan

plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})



# Rename columns 

plot_data = plot_data.rename(columns = {'Site EUI (kBtu/ft²)': 'Site EUI', 

                                        'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',

                                        'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})



# Drop na values

plot_data = plot_data.dropna()



# Function to calculate correlation coefficient between two columns

def corr_func(x, y, **kwargs):

    r = np.corrcoef(x, y)[0][1]

    ax = plt.gca()

    ax.annotate("r = {:.2f}".format(r),

                xy=(.2, .8), xycoords=ax.transAxes,

                size = 20)



# Create the pairgrid object

grid = sns.PairGrid(data = plot_data, size = 3)



# Upper is a scatter plot

grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)



# Diagonal is a histogram

grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')



# Bottom is correlation and density plot

grid.map_lower(corr_func);

grid.map_lower(sns.kdeplot, cmap = plt.cm.Reds)



# Title for entire plot

plt.suptitle('Pairs Plot of Energy Data', size = 36, y = 1.02);
# Copy the original data

features = data.copy()



# Select the numeric columns

numeric_subset = data.select_dtypes('number')



# Create columns with log of numeric columns

for col in numeric_subset.columns:

    # Skip the Energy Star Score column

    if col == 'score':

        next

    else:

        numeric_subset['log_' + col] = np.log(numeric_subset[col])

        

# Select the categorical columns

categorical_subset = data[['Borough', 'Largest Property Use Type']]



# One hot encode

categorical_subset = pd.get_dummies(categorical_subset)



# Join the two dataframes using concat

# Make sure to use axis = 1 to perform a column bind

features = pd.concat([numeric_subset, categorical_subset], axis = 1)



features.shape
plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()



plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')

plt.xlabel('Site EUI'); plt.ylabel('Weather Norm EUI')

plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' % np.corrcoef(data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna(), rowvar=False)[0][1]);
def remove_collinear_features(x, threshold):

    '''

    Objective:

        Remove collinear features in a dataframe with a correlation coefficient

        greater than the threshold. Removing collinear features can help a model

        to generalize and improves the interpretability of the model.

        

    Inputs: 

        threshold: any features with correlations greater than this value are removed

    

    Output: 

        dataframe that contains only the non-highly-collinear features

    '''

    

    # Dont want to remove correlations between Energy Star Score

    y = x['score']

    x = x.drop(columns = ['score'])

    

    # Calculate the correlation matrix

    corr_matrix = x.corr()

    iters = range(len(corr_matrix.columns) - 1)

    drop_cols = []

    

    # Iterate through the correlation matrix and compare correlations

    for i in iters:

        for j in range(i):

            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]

            col = item.columns

            row = item.index

            val = abs(item.values)

            

            # If correlation exceeds the threshold

            if val >= threshold:

                # Print the correlated features and the correlation value

                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))

                drop_cols.append(col.values[0])



    # Drop one of each pair of correlated columns

    drops = set(drop_cols)

    x = x.drop(columns = drops)

    x = x.drop(columns = ['Weather Normalized Site EUI (kBtu/ft²)', 

                          'Water Use (All Water Sources) (kgal)',

                          'log_Water Use (All Water Sources) (kgal)',

                          'Largest Property Use Type - Gross Floor Area (ft²)'])

    

    # Add the score back in to the data

    x['score'] = y

               

    return x
# Remove the collinear features above a specified correlation coefficient

features = remove_collinear_features(features, 0.6);
# Remove any columns with all na values

features  = features.dropna(axis=1, how = 'all')

features.shape
features.head()
# Extract the buildings with no score and the buildings with a score

no_score = features[features['score'].isna()]

score = features[features['score'].notnull()]



print(no_score.shape)

print(score.shape)
# Splitting data into training and testing

from sklearn.model_selection import train_test_split



# Separate out the features and targets

features = score.drop(columns='score')

targets = pd.DataFrame(score['score'])



# Replace the inf and -inf with nan (required for later imputation)

features = features.replace({np.inf: np.nan, -np.inf: np.nan})



# Split into 70% training and 30% testing set

X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)



print(X.shape)

print(X_test.shape)

print(y.shape)

print(y_test.shape)
# Function to calculate mean absolute error

def mae(y_true, y_pred):

    return np.mean(abs(y_true - y_pred))
baseline_guess = np.median(y)



print('The baseline guess is a score of %0.2f' % baseline_guess)

print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))
# Save the no scores, training, and testing data

no_score.to_csv('no_score.csv', index = False)

X.to_csv('training_features.csv', index = False)

X_test.to_csv('testing_features.csv', index = False)

y.to_csv('training_labels.csv', index = False)

y_test.to_csv('testing_labels.csv', index = False)
# Pandas and numpy for data manipulation

import pandas as pd

import numpy as np



# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 60)



# Matplotlib for visualization

import matplotlib.pyplot as plt



# Set default font size

plt.rcParams['font.size'] = 24



from IPython.core.pylabtools import figsize



# Seaborn for visualization

import seaborn as sns

sns.set(font_scale = 2)



# Imputing missing values and scaling values

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer as Imputer



# Machine Learning Models

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



# Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# Read in data into dataframes 

train_features = pd.read_csv('training_features.csv')

test_features = pd.read_csv('testing_features.csv')

train_labels = pd.read_csv('training_labels.csv')

test_labels = pd.read_csv('testing_labels.csv')



# Display sizes of data

print('Training Feature Size: ', train_features.shape)

print('Testing Feature Size:  ', test_features.shape)

print('Training Labels Size:  ', train_labels.shape)

print('Testing Labels Size:   ', test_labels.shape)
train_features.head(12)
figsize(8, 8)



# Histogram of the Energy Star Score

plt.style.use('fivethirtyeight')

plt.hist(train_labels['score'].dropna(), bins = 100);

plt.xlabel('Score'); plt.ylabel('Number of Buildings'); 

plt.title('ENERGY Star Score Distribution');
# Create an imputer object with a median filling strategy

imputer = Imputer(strategy='median')



# Train on the training features

imputer.fit(train_features)



# Transform both training data and testing data

X = imputer.transform(train_features)

X_test = imputer.transform(test_features)
print('Missing values in training features: ', np.sum(np.isnan(X)))

print('Missing values in testing features:  ', np.sum(np.isnan(X_test)))
# Make sure all values are finite

print(np.where(~np.isfinite(X)))

print(np.where(~np.isfinite(X_test)))
# Create the scaler object with a range of 0-1

scaler = MinMaxScaler(feature_range=(0, 1))



# Fit on the training data

scaler.fit(X)



# Transform both the training and testing data

X = scaler.transform(X)

X_test = scaler.transform(X_test)
# Convert y to one-dimensional array (vector)

y = np.array(train_labels).reshape((-1, ))

y_test = np.array(test_labels).reshape((-1, ))
# Function to calculate mean absolute error

def mae(y_true, y_pred):

    return np.mean(abs(y_true - y_pred))



# Takes in a model, trains the model, and evaluates the model on the test set

def fit_and_evaluate(model):

    

    # Train the model

    model.fit(X, y)

    

    # Make predictions and evalute

    model_pred = model.predict(X_test)

    model_mae = mae(y_test, model_pred)

    

    # Return the performance metric

    return model_mae
lr = LinearRegression()

lr_mae = fit_and_evaluate(lr)



print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)
svm = SVR(C = 1000, gamma = 0.1)

svm_mae = fit_and_evaluate(svm)



print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)
random_forest = RandomForestRegressor(random_state=60)

random_forest_mae = fit_and_evaluate(random_forest)



print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)
gradient_boosted = GradientBoostingRegressor(random_state=60)

gradient_boosted_mae = fit_and_evaluate(gradient_boosted)



print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)
knn = KNeighborsRegressor(n_neighbors=10)

knn_mae = fit_and_evaluate(knn)



print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)
plt.style.use('fivethirtyeight')

figsize(8, 6)



# Dataframe to hold the results

model_comparison = pd.DataFrame({'model': ['Linear Regression', 'Support Vector Machine',

                                           'Random Forest', 'Gradient Boosted',

                                            'K-Nearest Neighbors'],

                                 'mae': [lr_mae, svm_mae, random_forest_mae, 

                                         gradient_boosted_mae, knn_mae]})



# Horizontal bar chart of test mae

model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',

                                                           color = 'red', edgecolor = 'black')



# Plot formatting

plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)

plt.title('Model Comparison on Test MAE', size = 20);
# Loss function to be optimized

loss = ['ls', 'lad', 'huber']



# Number of trees used in the boosting process

n_estimators = [100, 500, 900, 1100, 1500]



# Maximum depth of each tree

max_depth = [2, 3, 5, 10, 15]



# Minimum number of samples per leaf

min_samples_leaf = [1, 2, 4, 6, 8]



# Minimum number of samples to split a node

min_samples_split = [2, 4, 6, 10]



# Maximum number of features to consider for making splits

max_features = ['auto', 'sqrt', 'log2', None]



# Define the grid of hyperparameters to search

hyperparameter_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf

                      }
# Create the model to use for hyperparameter tuning

model = RandomForestRegressor(random_state = 42)



# Set up the random search with 4-fold cross validation

random_cv = RandomizedSearchCV(estimator = model, 

                               param_distributions = hyperparameter_grid, 

                               n_iter = 25, 

                               cv = 4,

                               scoring = 'neg_mean_absolute_error',

                               verbose=2, 

                               random_state=42, 

                               n_jobs = -1)
# Fit on the training data

random_cv.fit(X, y)
# Get all of the cv results and sort by the test performance

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)



random_results.head(10)
random_cv.best_estimator_
# Create a range of trees to evaluate

trees_grid = {'n_estimators': [650, 700, 750, 800, 850, 900, 950, 1000]}



model = RandomForestRegressor(max_depth = 15,

                                  min_samples_leaf = 2,

                                  min_samples_split = 6,

                                  max_features = None,

                                  random_state = 42)



# Grid Search Object using the trees range and the random forest model

grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4, 

                           scoring = 'neg_mean_absolute_error', verbose = 1,

                           n_jobs = -1, return_train_score = True)
# Fit the grid search

grid_search.fit(X, y)
# Get the results into a dataframe

results = pd.DataFrame(grid_search.cv_results_)



# Plot the training and testing error vs number of trees

figsize(8, 8)

plt.style.use('fivethirtyeight')

plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label = 'Testing Error')

plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label = 'Training Error')

plt.xlabel('Number of Trees'); plt.ylabel('Mean Abosolute Error'); plt.legend();

plt.title('Performance vs Number of Trees');
results.sort_values('mean_test_score', ascending = False).head(5)
# Default model

default_model = RandomForestRegressor(random_state = 42)



# Select the best model

final_model = grid_search.best_estimator_



final_model
%%timeit -n 1 -r 5

default_model.fit(X, y)
%%timeit -n 1 -r 5

final_model.fit(X, y)
default_pred = default_model.predict(X_test)

final_pred = final_model.predict(X_test)



print('Default model performance on the test set: MAE = %0.4f.' % mae(y_test, default_pred))

print('Final model performance on the test set:   MAE = %0.4f.' % mae(y_test, final_pred))
figsize(8, 8)



# Density plot of the final predictions and the test values

sns.kdeplot(final_pred, label = 'Predictions')

sns.kdeplot(y_test, label = 'Values')



# Label the plot

plt.xlabel('Energy Star Score'); plt.ylabel('Density');

plt.title('Test Values and Predictions');
figsize = (6, 6)



# Calculate the residuals 

residuals = final_pred - y_test



# Plot the residuals in a histogram

plt.hist(residuals, color = 'red', bins = 20,

         edgecolor = 'black')

plt.xlabel('Error'); plt.ylabel('Count')

plt.title('Distribution of Residuals');
from sklearn import tree



# LIME for explaining predictions

import lime 

import lime.lime_tabular
# Extract the feature importances into a dataframe

feature_results = pd.DataFrame({'feature': list(train_features.columns), 

                                'importance': final_model.feature_importances_})



# Show the top 10 most important

feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)



feature_results.head(10)
figsize = (12, 10)

plt.style.use('fivethirtyeight')



# Plot the 10 most important features in a horizontal bar chart

feature_results.loc[:9, :].plot(x = 'feature', y = 'importance', 

                                 edgecolor = 'k',

                                 kind='barh', color = 'blue');

plt.xlabel('Relative Importance', size = 20); plt.ylabel('')

plt.title('Feature Importances from Random Forest', size = 30);
# Extract the names of the most important features

most_important_features = feature_results['feature'][:10]



# Find the index that corresponds to each feature name

indices = [list(train_features.columns).index(x) for x in most_important_features]



# Keep only the most important features

X_reduced = X[:, indices]

X_test_reduced = X_test[:, indices]



print('Most important training features shape: ', X_reduced.shape)

print('Most important testing  features shape: ', X_test_reduced.shape)
lr = LinearRegression()



# Fit on full set of features

lr.fit(X, y)

lr_full_pred = lr.predict(X_test)



# Fit on reduced set of features

lr.fit(X_reduced, y)

lr_reduced_pred = lr.predict(X_test_reduced)



# Display results

print('Linear Regression Full Results: MAE =    %0.4f.' % mae(y_test, lr_full_pred))

print('Linear Regression Reduced Results: MAE = %0.4f.' % mae(y_test, lr_reduced_pred))
# Create the model with the same hyperparamters

model_reduced = GradientBoostingRegressor(loss='lad', max_depth=5, max_features=None,

                                  min_samples_leaf=6, min_samples_split=6, 

                                  n_estimators=800, random_state=42)



# Fit and test on the reduced set of features

model_reduced.fit(X_reduced, y)

model_reduced_pred = model_reduced.predict(X_test_reduced)



print('Gradient Boosted Reduced Results: MAE = %0.4f' % mae(y_test, model_reduced_pred))
# Find the residuals

residuals = abs(model_reduced_pred - y_test)

    

# Exact the worst and best prediction

wrong = X_test_reduced[np.argmax(residuals), :]

right = X_test_reduced[np.argmin(residuals), :]
# Create a lime explainer object

explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_reduced, 

                                                   mode = 'regression',

                                                   training_labels = y,

                                                   feature_names = list(most_important_features))
# Display the predicted and true value for the wrong instance

print('Prediction: %0.4f' % model_reduced.predict(wrong.reshape(1, -1)))

print('Actual Value: %0.4f' % y_test[np.argmax(residuals)])



# Explanation for wrong prediction

wrong_exp = explainer.explain_instance(data_row = wrong, 

                                       predict_fn = model_reduced.predict)



# Plot the prediction explaination

wrong_exp.as_pyplot_figure();

plt.title('Explanation of Prediction', size = 28);

plt.xlabel('Effect on Prediction', size = 22);
wrong_exp.show_in_notebook(show_predicted_value=False)
# Display the predicted and true value for the wrong instance

print('Prediction: %0.4f' % model_reduced.predict(right.reshape(1, -1)))

print('Actual Value: %0.4f' % y_test[np.argmin(residuals)])



# Explanation for wrong prediction

right_exp = explainer.explain_instance(right, model_reduced.predict, num_features=10)

right_exp.as_pyplot_figure();

plt.title('Explanation of Prediction', size = 28);

plt.xlabel('Effect on Prediction', size = 22);
right_exp.show_in_notebook(show_predicted_value=False)
# Extract a single tree

single_tree = model_reduced.estimators_[105][0]



tree.export_graphviz(single_tree, out_file = 'tree.dot',

                     rounded = True, 

                     feature_names = most_important_features,

                     filled = True,

                    max_depth = 3)



single_tree
# Convert to a png

import pydot



(graph,) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')
from IPython.display import Image

Image("tree.png")