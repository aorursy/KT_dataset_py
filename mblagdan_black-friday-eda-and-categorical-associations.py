# Loading the essentials, numpy and pandas



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import scipy.stats as ss



# Next up, os for listing and walking through directories

import os



# Plt and seaborn for graphing

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable

import seaborn as sns



print(os.listdir("../input"))
df_blackfriday = pd.read_csv("../input/BlackFriday.csv")



df_blackfriday.head()
nan_to_total_ratio = df_blackfriday.isna().sum() / df_blackfriday.shape[0]

print(nan_to_total_ratio)
# Save these for later use

product_category_2_series = df_blackfriday['Product_Category_2']

product_category_3_series = df_blackfriday['Product_Category_3']



df_blackfriday = df_blackfriday.drop(['Product_Category_2', 'Product_Category_3'], axis=1)



# Store the relevant categories

categories = df_blackfriday.columns[2:-1]



# Define function for our check

# Function used so I can easily exit two nested loops with the 'return' keyword

def check_duplicates_and_differences(category, groupby_object):

    # Iterate over a GroupBy object, where keys are User IDs,

    # and groups are groups of categorical values sharing a User ID

    for key,group in groupby_object:

        value_list = []

        # Iterate over individual values in group

        for i, value in enumerate(group):

            # Branch code here: in the case of Product ID, we check for existence of duplicate values,

            # else we check for differences between current and previous value

            if category == 'Product_ID':

                if value in value_list:

                    print("Found duplicate value: {0} in user_id: {1} of category: {2}".format(value,group,category))

                    # Break the loops if any duplicates are found within a category

                    return

                value_list.append(value)

            else:

                if i>=1 and value != value_list[i-1]:

                    print("Variable {0} of user {1} changes from {2} to {3}".format(category, key, value_list[i-1], value))

                    # Break the loops if any changes are found within a category

                    return

                value_list.append(value)



for category in categories:

    # First, group our category by User ID

    grouped = df_blackfriday[category].groupby(df_blackfriday['User_ID'])



    check_duplicates_and_differences(category, grouped)
ncols = 2



# Assuming ncols, calculate number of rows based on number of categories times two

# since we have 2 graphs per cat, then use these values for our grid size

nrows = math.ceil(len(categories)*2/ncols)

grid_size = (nrows,ncols)

print(grid_size)

# Multiplier to convert grid size to appropriate size in inches

inch_multiplier = 4



grid_inches = tuple(map(lambda x: x * inch_multiplier, grid_size))



plt.figure(figsize=grid_inches)



# 2-step iteration over number of categories times two

for index,category in zip(range(1, len(category)*2, 2), categories):

    plt.subplot(grid_size[0],grid_size[1],index)

    # Create a common color mapping for both cases so we can more easily compare the differences

    subcategories = df_blackfriday[category].unique()

    rgb_values = sns.color_palette("Set1", len(subcategories))

    color_map = dict(zip(subcategories, rgb_values))

    

    # First, plot distribution of users for category

    if not category == 'Product_Category_1':

        # Group by user and aggregate values by replacing them with the first value - can do this since

        # I've shown that the values in these categories don't change per user

        df_grouped_by_user = df_blackfriday[category].groupby(df_blackfriday['User_ID']).agg('first')

        # Use normalize parameter to obtain fractions

        df_grouped_by_user = df_grouped_by_user.sort_values().value_counts(normalize=True)

        df_grouped_by_user.plot(kind='bar', title=category, color=df_grouped_by_user.index.map(color_map))

        plt.ylabel('No. of users')

    else:

        df_grouped = df_blackfriday[category].sort_values().value_counts(normalize=True)

        df_grouped.plot(kind='bar', title=category, color=df_grouped.index.map(color_map))



    # Second, plot amount of purchases in dollars per category

    plt.subplot(grid_size[0],grid_size[1],index+1)

    

    df_grouped_by_cat = df_blackfriday['Purchase'].groupby(df_blackfriday[category]).agg('sum')

    # Divide Series elements by sum of all purchases to get fractions that we can 

    # compare with normalized count distributions

    df_grouped_by_cat = df_grouped_by_cat.divide(df_blackfriday['Purchase'].sum())

    df_grouped_by_cat = df_grouped_by_cat.sort_values(ascending=False)

    df_grouped_by_cat.plot(kind='bar', title=category, color=df_grouped_by_cat.index.map(color_map))

    plt.ylabel('Purchases')

        

plt.subplots_adjust(wspace = 0.2, hspace = 0.5, top=3)
def cramers_corrected_stat(confusion_matrix):

    """ calculate Cramers V statistic for categorical-categorical association.

        uses correction from Bergsma and Wicher, 

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    # Use sum twice because of the way it works for dataframes

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))



def correlation_ratio(dataframe, nominal_series_name, numerical_series_name):

    categories_means = []

    categories_weights = []

    total_mean = np.average(dataframe[numerical_series_name])

    total_variance = np.var(dataframe[numerical_series_name])

    for category in dataframe[nominal_series_name].unique():

        category_series = dataframe.loc[dataframe[nominal_series_name] == category][numerical_series_name]

        category_mean = np.average(category_series)

        categories_means.append(category_mean)

        categories_weights.append(len(category_series))



    categories_weighted_variance = np.average((categories_means - total_mean)**2, weights=categories_weights)

    eta = categories_weighted_variance / total_variance

    return eta
def create_corr_matrix(dataframe, nominal_columns, numerical_columns):

    columns = dataframe.columns

    # Forcing dtype np.float64 seems to be important for sns.heatmap() method to work with the output correlation matrix

    corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=np.float64)

    for i in range(0, len(columns)):

        for j in range(i, len(columns)):

            if i == j:

                corr_matrix.at[columns[j], columns[i]] = 1.00

            else:

                if columns[i] in nominal_columns:

                    if columns[j] in nominal_columns:

                        # Categorical to categorical correlation

                        confusion_matrix = pd.crosstab(dataframe[columns[i]], dataframe[columns[j]])

                        corr_coef = cramers_corrected_stat(confusion_matrix)

                        corr_matrix.at[columns[j], columns[i]] = corr_coef

                        corr_matrix.at[columns[i], columns[j]] = corr_coef

                    else:

                        # Categorical to continuous correlation

                        corr_coef = correlation_ratio(dataframe,columns[i], columns[j])

                        corr_matrix.at[columns[j], columns[i]] = corr_coef

                        corr_matrix.at[columns[i], columns[j]] = corr_coef

                else:

                    if columns[j] in nominal_columns:

                        # Continuous to categorical correlation

                        corr_coef = correlation_ratio(dataframe, columns[j], columns[i])

                        corr_matrix.at[columns[j], columns[i]] = corr_coef

                        corr_matrix.at[columns[i], columns[j]] = corr_coef

                    else:

                        # Continuous to continuous correlation - using Spearman coefficient here

                        corr_coef, pval = ss.spearmanr(dataframe[columns[j]], dataframe[columns[i]])

                        corr_matrix.at[columns[j], columns[i]] = corr_coef

                        corr_matrix.at[columns[i], columns[j]] = corr_coef

    return corr_matrix   
# Drop User_ID, Product_ID and our only continuous variable, Purchase

nominal_columns = df_blackfriday.columns.drop(['User_ID','Product_ID','Purchase'])

numerical_columns = ['Purchase']

df_blackfriday_nouser = df_blackfriday.drop(['User_ID', 'Product_ID'], axis=1)

corr_matrix = create_corr_matrix(df_blackfriday_nouser, nominal_columns, numerical_columns)



plt.subplots(figsize=(18,8))



# Use vmin = 0 since we only have one numerical column, and the non-continuous association measures used

# here range from 0 to 1

sns.heatmap(corr_matrix, vmin=0, square=True)