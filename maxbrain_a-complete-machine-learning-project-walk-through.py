# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None



# Display up to 60 columns of a dataframe

pd.set_option('display.max_columns', 60)



# Matplotlib visualization

import matplotlib.pyplot as plt

%matplotlib inline



# Set default font size

plt.rcParams['font.size'] = 24



# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize



# Seaborn for visualization

import seaborn as sns

sns.set(font_scale = 2)



# Splitting data into training and testing

from sklearn.model_selection import train_test_split
data = pd.read_excel('/kaggle/input/nyc_benchmarking_disclosure_2017_consumption_data.xlsx','Information and Metrics')

data.head()

# See the column data types and non-missing values

data.info()
# Replace all occurrences of Not Available with numpy not a number

data = data.replace({'Not Available': np.nan})
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
data.loc[data['Site EUI (kBtu/ft²)'] == 860, :]
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
# Create a list of buildings with more than 100 measurements

types = data.dropna(subset=['score'])

types = types['Largest Property Use Type'].value_counts()

types = list(types[types.values > 100].index)
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
# Extract the buildings with no score and the buildings with a score

no_score = features[features['score'].isna()]

score = features[features['score'].notnull()]



print(no_score.shape)

print(score.shape)
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