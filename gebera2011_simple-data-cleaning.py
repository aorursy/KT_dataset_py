import pandas as pd

import numpy as np



pd.set_option('display.max_columns', 60)

import matplotlib.pyplot as plt

%matplotlib inline



from IPython.core.pylabtools import figsize

# Set default font size

plt.rcParams['font.size'] = 24



import seaborn as sns

sns.set(font_scale = 2)



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.shape, test.shape
data = pd.concat([train, test], sort=False)

data.shape
data.head()
data.tail()
data.info()
for col in list(data.columns):   

    if ('Class' in col):

        data[col] = data[col].astype(str)    
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
missing_df = missing_values_table(data)

missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)

print('We will remove %d columns.' % len(missing_columns))
data = data.drop(columns=missing_columns)
figsize(8, 8)

plt.style.use('fivethirtyeight')

plt.hist(data['SalePrice'].dropna(), bins = 100, edgecolor = 'k')

plt.xlabel('Sale Price'); plt.ylabel('Number of houses')

plt.title('Sale Price Distribution')
# from the figure, we see outliers at the right end

data['SalePrice'].describe()
data['SalePrice'].dropna().sort_values().tail(15)
data.loc[data['SalePrice'] > 700000]
# calculate the first and third quartile

first_quartile = data['SalePrice'].describe()['25%']

third_quartile = data['SalePrice'].describe()['75%']

Interquartile = third_quartile - first_quartile



upper_outlier = data.loc[data['SalePrice'] > third_quartile + 3 * Interquartile].index

lower_outlier = data.loc[data['SalePrice'] < first_quartile - 3 * Interquartile].index

len(list(upper_outlier) + list(lower_outlier))
# calculate the first and third quartile

first_quartile = data['GrLivArea'].describe()['25%']

third_quartile = data['GrLivArea'].describe()['75%']

Interquartile = third_quartile - first_quartile



upper_outlier = data.loc[data['GrLivArea'] > third_quartile + 3 * Interquartile].index

lower_outlier = data.loc[data['GrLivArea'] < first_quartile - 3 * Interquartile].index

len(list(upper_outlier) + list(lower_outlier))
data['GrLivArea'].dropna().sort_values().tail(15)
# Remove the two outliers

data.loc[data['GrLivArea'] > 4500, ['GrLivArea', 'SalePrice']]
data = data.loc[(data['GrLivArea'] < 4500) | (data['SalePrice'].isna())]
Zones = data.dropna(subset=['SalePrice'])

Zones = Zones['MSZoning'].value_counts()

Zones = Zones[Zones>50].index
'''

Plot of distribution of Sale Price for zone class 

       A	Agriculture

       C	Commercial

       FV	Floating Village Residential

       I	Industrial

       RH	Residential High Density

       RL	Residential Low Density

       RP	Residential Low Density Park 

       RM	Residential Medium Density

'''

figsize(12, 10)



# Plot each building

for zone in Zones:

    # Select the zone type

    subset = data[data['MSZoning'] == zone]

    

    # Density plot of Sale Price

    sns.kdeplot(subset['SalePrice'].dropna(),

               label = zone, shade = False, alpha = 0.8);

    

# label the plot

plt.xlabel('Sale Price', size = 20); plt.ylabel('Density', size = 20); 

plt.title('Density Plot of Sale Price by general zone classification', size = 28);
Nbos = data.dropna(subset=['SalePrice'])

Nbos = Nbos['Neighborhood'].value_counts()

Nbos = Nbos[Nbos>50].index
# Plot of distribution of Sale Price for neighborhoods

figsize(12, 10)



for Nbo in Nbos:    

    subset = data[data['Neighborhood'] == Nbo]

    

    # Density plot of Sale Price

    sns.kdeplot(subset['SalePrice'].dropna(),

               label = Nbo, shade = False, alpha = 0.8);

    

# label the plot

plt.xlabel('Sale Price', size = 20); plt.ylabel('Density', size = 20); 

plt.title('Density Plot of Sale Price by Neighborhoods', size = 28);
# Find all correlations and sort 

correlations_data = data.corr()['SalePrice'].sort_values()



# Print the most negative correlations

print(correlations_data.head(15), '\n')



# Print the most positive correlations

print(correlations_data.tail(15))
# Select the numeric columns

num_features = data.select_dtypes('number')



# Create columns with square root and log of numeric columns

for col in num_features.columns:

    # Skip the target column

    if col == 'SalePrice':

        next

    else:

        num_features['sqrt_' + col] = np.sqrt(num_features[col])

        num_features['log_' + col] = np.log(num_features[col])



# Drop houses without sale price

num_features = num_features.dropna(subset = ['SalePrice'])



# Find correlations with the sale price 

correlations = num_features.corr()['SalePrice'].dropna().sort_values()
# Display most negative correlations

correlations.head(15)
correlations.tail(15)
# Select the categorical columns

categorical_subset = data.select_dtypes('object')



# One hot encode

cat_features = pd.get_dummies(categorical_subset)



cat_features['SalePrice'] = data['SalePrice']



# Find correlations with the sale price 

cat_features = cat_features.dropna(subset=['SalePrice'])

correlations = cat_features.corr()['SalePrice'].dropna().sort_values()
correlations.head(10)
correlations.tail(20)
'''

a rule of thumb for correlation

0.00-0.19: very weak

0.20-0.39: weak

0.40-0.59: moderate 

0.60-0.79: strong

0.80-1.00: very strong

here I choose cutoff values ca. > 0.3 or < -0.3

'''

cat_features = list(list(correlations.head(10).index)+list(correlations.tail(20).index))

cat_features_names = list()

for feature in cat_features:

    cat_features_names.append(feature.split('_')[0])

    

cat_features_names = list(set(cat_features_names))

cat_features_names 
cat_features = data[cat_features_names]

cat_features = cat_features.dropna(subset=['SalePrice'])
cat_features = cat_features.drop(columns=['SalePrice'])

cat_features_names.remove('SalePrice')

features = pd.concat([cat_features, num_features],axis=1)

features.columns
figsize(12, 10)



# Limit to zone types with more than 50 observations (from previous code)

features = features[features['MSZoning'].isin(Zones)]



# Use seaborn to plot a scatterplot of Sale Price vs. Overall Quality

sns.lmplot('OverallQual', 'SalePrice', 

          hue = 'MSZoning', data = features,

          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,

          size = 12, aspect = 1.2);



# Plot labeling

plt.xlabel("Overall Quality", size = 28)

plt.ylabel('Sale Price', size = 28)

plt.title('Overall Quality vs. Sale Price', size = 36);
figsize(12, 10)

# Limit to neighborhood with more than 50 observations (from previous code)

features = features[features['Neighborhood'].isin(Nbos)]



# Use seaborn to plot a scatterplot of sale price vs overall quality

sns.lmplot('OverallQual', 'SalePrice', 

          hue = 'Neighborhood', data = features,

          scatter_kws = {'alpha': 0.8, 's': 60}, fit_reg = False,

          size = 12, aspect = 1.2);



# Plot labeling

plt.xlabel("Overall Quality", size = 28)

plt.ylabel('Sale Price', size = 28)

plt.title('Overall Quality vs. Sale Price', size = 36);
# Extract the columns to  plot

plot_data = features[['SalePrice', 'OverallQual', 

                      'GrLivArea', 

                      'TotalBsmtSF']]



# Replace the inf with nan

plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})



# Rename columns 

plot_data = plot_data.rename(columns = {'SalePrice': 'Sale Price', 

                                        'OverallQual': 'Overall Quality',

                                        'GrLivArea': 'living area SF',

                                        'TotalBsmtSF':'Basement SF'})



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

plt.suptitle('Pairs Plot of Sale Price', size = 36, y = 1.02);
# Copy the original data

features = data.copy()



# Select the numeric columns

numeric_subset = data.select_dtypes('number')



# Create columns with log of numeric columns

for col in numeric_subset.columns:

    # Skip the Sale Price column

    if col == 'SalePrice':

        next

    else:

        numeric_subset['log_' + col] = np.log(numeric_subset[col]+0.0001)

        

# Select the categorical columns

categorical_subset = data[cat_features_names]



# fill missing category data before one hot encoding

categorical_subset = categorical_subset.fillna(categorical_subset.mode().iloc[0])



# One hot encode

categorical_subset = pd.get_dummies(categorical_subset)



# Join the two dataframes using concat

# Make sure to use axis = 1 to perform a column bind

features = pd.concat([numeric_subset, categorical_subset], axis = 1)



features.shape
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

    

    # Dont want to remove correlations between Sale Price

    y = x['SalePrice']

    x = x.drop(columns = ['SalePrice'])

    

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

    

    

    # Add the target back in to the data

    x['SalePrice'] = y

               

    return x
# Remove the collinear features above a specified correlation coefficient

features = remove_collinear_features(features, 0.6);
# Remove any columns with all na values

features  = features.dropna(axis=1, how = 'all')

features.shape
no_price = features[features['SalePrice'].isna()]

price = features[features['SalePrice'].notnull()]



print(no_price.shape)

print(price.shape)
# Separate out the features and targets

features = price.drop(columns='SalePrice')

targets = pd.DataFrame(price['SalePrice'])



# Replace the inf and -inf with nan (required for later imputation)

features = features.replace({np.inf: np.nan, -np.inf: np.nan})



# Split into 80% training and 20% testing set

X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)



print(X.shape)

print(X_test.shape)

print(y.shape)

print(y_test.shape)
# Function to calculate mean absolute error

def mae(y_true, y_pred):

    return np.mean(abs(y_true - y_pred))
baseline_guess = np.median(y)



print('The baseline guess of median price %0.2f' % baseline_guess)

print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))