# To begin this exploratory analysis, first import libraries and define functions and utilities to work with the data.



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for beautiful plots and some types of graphs

import seaborn as sns
# IMPORTING THE DATA

# There is 1 csv file in the current version of the dataset



kBaseDataDirectory = "/kaggle/input"  # on Kaggle

#kBaseDataDirectory = "./kaggle/input"  # when working offline with jupyter notebook



dataset_files = []



# This loop will import all dataset files in case we add more data in a next version of the dataset

for dirname, _, filenames in os.walk(kBaseDataDirectory):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dataset_files.append(os.path.join(dirname, filename))

#######   Utility functions for statistics   #######



### Helpers to filter dataframes



def helper_has_fields_compared_to(df, columns, target, what, operator):

    """

    Helper to compare several columns to the same value.

    """

    col = columns[0]

    res = df[col] > target

    for col in columns[1:]:

        if operator == '>':

            tmp = (df[col] > target)

        elif operator == '>=':

            tmp = (df[col] >= target)

        elif operator == '<=':

            tmp = (df[col] <= target)

        elif operator == '<':

            tmp = (df[col] < target)

        elif operator == '==':

            tmp = (df[col] == target)

        elif operator == '!=':

            tmp = (df[col] != target)

        

        # 

        if what == 'all':

            res = res & tmp

        elif what in ['any']:

            res = res | tmp

    return res



def helper_has_any_field_greater_than(df, columns, target):

    """Returns lines of the dataframe where any of value of the specified columns

    is greater than the target.

    """

    res = helper_has_fields_compared_to(df, columns, target, 'any', '>')

    return res



def helper_has_all_field_greater_than(df, columns, target):

    res = helper_has_fields_compared_to(df, columns, target, 'all', '>')

    return res





### Other utilities for stats



def frequency(data, probabilities=False, sort=False, reverse=False):

    """Returns the frequency distribution of elements.

    This is a convenience method for effectif()'s most common use case, without all the more complicated parameters.

    :param data: A collection of elements you want to count.

    :param bool probabilities: Whether you want the result frequencies to sum up to 1. Default: False

    """

    xis, nis = effectif(data, returnSplitted=True, frequencies=probabilities, sort=sort, reverse=reverse)

    return xis, nis





def frequences(data, returnSplitted=True, hashAsString=False, universe=None, frequenciesOverUniverse=None):

    """

    """

    if universe is None:

        return effectif(data, returnSplitted, hashAsString, True)

    else:

        return effectifU(data, universe, returnSplitted, hashAsString, True, frequenciesOverUniverse)

    



def effectif(data, returnSplitted=True, hashAsString=False, frequencies=False, inputConverter=None, sort=False, reverse=False):

    """calcule l'effectif

    :param list data: une liste

    :param bool hashAsString: whether we should convert the values in 'data' to

                string before comparing them

    :param function inputConverter: a callable function that is used to convert

                the values within data into the class you want the values to be

                compared as. When not provided, the identity function is used.

                If used with parameter 'hashAsString', the hashed value will be

                the one returned by this function.

    :param bool sort: sort the result (only if returnSplitted). Shorthand for `sortBasedOn`

    :param bool reverse: reverse the order (only if sort and returnSplitted). Shorthand for `sortBasedOn`

    """

    inputConverter = (lambda x: x) if inputConverter is None else inputConverter

    effs = {}

    for val in data:

        val = inputConverter(val)

        key = str(val) if hashAsString else val

        try:

            effs[key] = effs[key]+1

        except:

            effs[key] = 1

    

    if frequencies:

        tot = sum(effs.values())

        for key in effs:

            effs[key] = effs[key]/tot

    

    if returnSplitted:

        xis = list(effs.keys())

        nis = list(effs.values())

        if sort:

            xis, nis = sortBasedOn(nis, xis, nis, reverse=reverse)

        return xis, nis

    

    return effs

# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()



# Correlation matrix

def plotCorrelationMatrix(df, graphWidth, segmentName=None):

    filename = segmentName if segmentName else getattr(df, "dataframeName", segmentName)

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()



# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = None  # integer: How many rows to read. specify 'None' if want to read whole file

# loading the 1 dataset file of the kernel

main_input_dataset_filepath = dataset_files[0]

df1 = pd.read_csv(os.path.join(kBaseDataDirectory, main_input_dataset_filepath), delimiter=',', nrows = nRowsRead)



df1.dataframeName = 'users.dataset.public.csv'

orig_df1 = df1.copy()

orig_df1.dataframeName = 'users.dataset.public.csv'



nRow, nCol = df1.shape

print(f'There are {nRow} rows (i.e. users) and {nCol} columns (i.e. possible features)')

print('\nThese columns are: \n{}'.format(" - ".join(list(df1.columns))))
#orig_df1.head(10)

orig_df1.sample(10)
#print('Columns: \n{}'.format(" - ".join(list(df1.columns))))



useless_columns = []

# unused metadata are dropped

useless_columns += ["identifierHash", "type"]



# Duplicate columns are dropped in favor of their siblings

useless_columns += ["seniority", "seniorityAsYears", "civilityGenderId", "country"]



columns_unused_in_correlations = ["hasProfilePicture", "daysSinceLastLogin"]



# Unless one specifically want to look at the difference between iOS users and Android users

# For instance, "Do iphone/iOS users buy more than Android users, since iPhones are more pricey ?"

# However, keep in mind you should study the average users, and this version of the dataset is too small

# to study and conclude straight. Contact the author of the dataset for more dataset entries ;)

unused_mobile_app_columns = ["hasIosApp", "hasAndroidApp"]



# this way we can have cleaner graphs if needed.

all_unused_columns = useless_columns + unused_mobile_app_columns + columns_unused_in_correlations



## Dropping columns inplace to conserve the df1.dataframeName member

df1.drop(useless_columns, axis=1, errors='ignore', inplace=True)

#df1.drop(all_unused_columns, axis=1, errors='ignore', inplace=True)



## Reordering few columns for better visualization and concordance

cols = df1.columns.tolist()

_i = cols.index("socialProductsLiked")

_ = cols.pop(_i); cols.insert(cols.index("productsWished"), "socialProductsLiked")

_i_country = cols.index("countryCode"); cols.pop(_i_country); cols.insert(cols.index("language")+1, "countryCode")

df1 = df1[cols]



#print(df1.daysSinceLastLogin.describe(), "\n\n", df1.seniority.describe())



print(f"New shape: {df1.shape[0]} users (rows), {df1.shape[1]} features (columns)\n")

print("Remaining columns: \n{}".format(" - ".join(list(df1.columns))))

df1.sample(5)
df1.describe()
### Hence, let's create specific subsets of users regarding all that.



### ACTIVE USERS ###

# By filtering out users that are completely passive from the dataframe

# (i.e. users who did not sell/upload/buy/like/wish/... any product), we can study the behavior of real active users.



# using df1 instead of orig_df1 in case we had removed some rows after the initial import



active_df = df1[helper_has_any_field_greater_than(df1, ['socialProductsLiked', 'productsListed', 'productsSold',

       'productsPassRate', 'productsWished', 'productsBought'], 0)]

active_df.dataframeName = "Active Users"





### BUYING/SELLING BEHAVIOUR ###



### Buyers

buyers_df = df1[df1.productsBought > 0]

#buyers_df.productsBought.describe()

buyers_df.dataframeName = "Buyers"



### Prospecting Sellers (has sold a product or is still trying do his/her first sale)

sellers_df = df1[(df1.productsListed > 0) | (df1.productsSold > 0)]

sellers_df.dataframeName = "Prospecting Sellers"



### Successful sellers (at least 1 product sold)

successful_sellers_df = df1[df1.productsSold > 0]

successful_sellers_df.dataframeName = "Successful sellers"





### SOCIAL INTERACTIONS ###



# Users using the social media features of the service

# Tracks anyone who willingly un/-subscribed to someone.

# And it also includes accounts that are either good (increased their followers)

#   or bad (even basic followers unsubscribed from them).

# Since each new account is automatically assigned 3 followers and 8 accounts to follow

# I filter out those who differ from these default account settings.

social_df = df1[ (df1['socialNbFollowers'] != 3) | (df1['socialNbFollows'] != 8) ]

#asocial_users_df = df1[ (df1['socialNbFollows'] < 8) ]



# Among those social users, filter only those active on products (selling, buying or wishing for articles, ...)

market_social_df = social_df[helper_has_any_field_greater_than(social_df, ['socialProductsLiked', 'productsListed', 'productsSold',

       'productsPassRate', 'productsWished', 'productsBought'], 0)]



#print(socialUsers.shape, activeSocialUsers.shape)

#socialUsers.head(20)





### RESULTS / INFOS

print(f"""Out of the {orig_df1.shape[0]} users of the dataset sample, there are:""")

print()

print(f"""- {active_df.shape[0]} active users ({100*active_df.shape[0]/df1.shape[0]:.3}%). Among these prospective buyers and sellers""")



print(f"""  - {active_df.shape[0] - sellers_df.shape[0]} are prospective buyers""")

print(f"""    among which {buyers_df.shape[0]} people actually bought products (at least 1)""")

print(f"""  - {sellers_df.shape[0]} are prospective sellers""")

print(f"""    among which {successful_sellers_df.shape[0]} are successful sellers (>= 1 product successfully sold)""")

print()

print(f"""- {social_df.shape[0]} people using social network features""")

print(f"""  such as following accounts or getting followers""")

print("\nNote that among the above number of sellers, some may act as buyers and vice-versa")
print(f"Active users: {active_df.shape[0]} records with {active_df.shape[1]} columns")

active_df.sample(12)
#### Adding columns



## Period in which the user has not completely dropped out (in nbr of days)

## TODO: check if enough data for the formula

# df1["activityDays"] = df1.apply(lambda row: (row["seniority"] - row["daysSinceLastLogin"]), axis=1)
#df1["seniorityAsMonths"] = df1["seniorityAsMonths"].apply(lambda x: int(x))



#print("Months of seniority of the users in the dataset: \n- {} months".format(" months\n- ".join(map(str, df1["seniorityAsMonths"].unique()))))



#df1.head(10) #["seniorityAsMonths"]

#df1.seniorityAsMonths.describe()
### Distribution graphs (histogram/bar graph) of sampled columns:



#plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8, "All users")
#correlation_df = df1.drop(["hasProfilePicture", "hasAndroidApp", "hasIosApp"], axis=1, errors="ignore")

correlation_df = active_df.drop(["hasProfilePicture", "hasAndroidApp", "hasIosApp"], axis=1, errors="ignore")

correlation_df.drop(["seniority", "daysSinceLastLogin"], axis=1, errors="ignore", inplace=True)

plotCorrelationMatrix(correlation_df, 8, "Active Users")

plotCorrelationMatrix(buyers_df.drop(all_unused_columns, axis=1, errors="ignore"), 8, "Buyers")



print(f"""In average, buyers buy {buyers_df.productsBought.sum() / buyers_df.shape[0] :.2f} products. Details are as follows:""")

buyers_df.productsBought.describe()
dropout_after = lambda df, dayMax: dayMax - df.daysSinceLastLogin

max_last_login = df1.daysSinceLastLogin.max()

#df1 = df1[df1.daysSinceLastLogin <= df1.seniority]



plt.title(f"User retention: how users drop out over time [all user segments]")

#plt.xlabel("(present) <-- # days since last login  --> (past)")

#sns.kdeplot(np.array(df1.seniority), shade=True)

plt.xlabel("(past) <-- # days before dropout  --> (present)")

plt.ylabel("density")

sns.kdeplot(np.array(dropout_after(df1, max_last_login)), shade=True)

plt.show()

plt.title("User retention - how people who bought at least one product dropout")

#plt.xlabel("(present) <-- # days since last login  --> (past)")

#sns.kdeplot(np.array(buyers_df.daysSinceLastLogin), shade=True)

plt.xlabel("(past) <=  # days before dropout  => (present)")

plt.ylabel("density")

sns.kdeplot(np.array(dropout_after(buyers_df, max_last_login)), shade=True)

sns.kdeplot(np.array(dropout_after(buyers_df, max_last_login)), shade=True)

plt.show()



plt.title("User retention - how sellers keep visiting a C2C site")

#plt.xlabel("(present) <-- # days since last login  --> (past)")

plt.xlabel("(past) <=  # days before dropout  => (present)")

plt.ylabel("density")

sns.kdeplot(np.array(dropout_after(sellers_df, max_last_login)), shade=True)

plt.show()



plt.title("User retention - buyers and sellers dropout curve")

sns.kdeplot(np.array(dropout_after(sellers_df,max_last_login)), shade=True)

sns.kdeplot(np.array(dropout_after(buyers_df,max_last_login)), shade=True)

plt.xlabel("(past) <=  # days before dropout  => (present)")

plt.ylabel("density")

plt.legend(["sellers", "buyers"])

plt.show()