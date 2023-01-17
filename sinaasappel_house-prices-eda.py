import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np              # linear algebra

import matplotlib.pyplot as plt # data visualisation

%matplotlib inline              

import seaborn as sns           # data visualisation
# Path of the file to read

train_file_path = '../input/train.csv'

test_file_path  = '../input/test.csv' 



# Fill in the line below to read the file into a variable home_data

train = pd.read_csv(train_file_path, index_col='Id')

test  = pd.read_csv(test_file_path, index_col='Id')
train.head()
test.head()
mystring = "The train data contains {0} rows (observations) and {1} columns (variables)."

print(mystring.format(train.shape[0], train.shape[1]))



mystring = "The test data contains {0} rows (observations) and {1} columns (variables)."

print(mystring.format(test.shape[0], test.shape[1]))
print(train['SalePrice'].describe())

print("median  ", train['SalePrice'].median())

print("Number of missings:", train['SalePrice'].isna().sum())
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
# Set the width and height of the two plots combined

f, axes = plt.subplots(1, 2, figsize=(14, 5))





# make distribution plot

sns.distplot(a=train['SalePrice'], ax=axes[0])

sns.distplot(a = np.log(train['SalePrice']), ax=axes[1])
#skewness and kurtosis

print("Skewness before log transformation: %f" % train['SalePrice'].skew())

print("Kurtosis before log transformation: %f" % train['SalePrice'].kurt())



print("Skewness after log transformation: %f" % np.log(train['SalePrice']).skew())

print("Kurtosis after log transformation: %f" % np.log(train['SalePrice']).kurt())
# get a list of column names

var_names = list(train.drop(columns = ['SalePrice']).columns) 



print("There are {0} featues that can be used to predict the Saleprice.".format(len(var_names)))
train.drop(columns = ['SalePrice']).info()
# Here I assume that columns containing text values (dtypes == 'object') are categorical features

s = (train.dtypes == 'object')

cat_vars = list(s[s].index)

# I manually checked for other categorical features containing numeric values

other_cat_vars = ['MSSubClass', 'OverallQual', 'OverallCond', ]

categorical_vars = cat_vars + other_cat_vars
# make empty dictionary

data = {}



# get number of unique values per variable

for i in categorical_vars:

    variables = i

    n_unique_values = len(train[i].unique().tolist())

    data[i] = (variables, n_unique_values)

    

# go from dictionary to pandas dataframe    

df_cat_vars = pd.DataFrame.from_dict(data, orient='index', columns=['variables','n_unique_values'])



# sort values by n_unique values

ordered_df = df_cat_vars.sort_values(by = ['n_unique_values'], ascending=True)



# make horizontal barchart to visualize results

plt.figure(figsize=(10,10))                                       #set size of figure to 10x10 inches

plt.barh(ordered_df.variables, ordered_df.n_unique_values)        #plot barchart

plt.xlabel('number of unique values')                             #name x-label

plt.title('The number of unique values per categorical variable') #give plot a title

plt.show()

# get number of missings per column in train and test set

mistrain = train.isnull().sum().to_frame()

mistest = test.isnull().sum().to_frame()



# give column the name nMissings

mistrain.columns = ['nMissings']

mistest.columns = ['nMissings']



# make new columns that gives information about missing percentage

mistrain['percMissing'] = mistrain['nMissings']/1460

mistest['percMissing'] = mistest['nMissings']/1459



# select only rows with nMissings >= 1

mistrain = mistrain[mistrain.nMissings >= 1]

mistest = mistest[mistest.nMissings >= 1]



# sort values by nMissings values

ordered_df_train = mistrain.sort_values(by = ['nMissings'], ascending=False)

ordered_df_test = mistest.sort_values(by = ['nMissings'], ascending=False)
# set figure size

plt.figure(figsize=(4,8))          



# add title

plt.title('Missings values in the train set')  



# add lable for horizontal axis

plt.xlabel('Percentage of missing values')                            



# make barplot

sns.barplot(y = ordered_df_train.index, x = ordered_df_train['percMissing'])   
# set figure size

plt.figure(figsize=(4,8))          



# add title

plt.title('Missings values in the test set')  



# add lable for horizontal axis

plt.xlabel('Percentage of missing values')



# make barplot

sns.barplot(y=ordered_df_test.index, x=ordered_df_test['percMissing'])      
# select target variable SalePrice and features related to size

my_vars = ['SalePrice', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']



# get the correlation coefficients between these features.

corr = train[my_vars].corr()
# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Correlation between each feature related to size and SalePrice")



# Heatmap showing average arrival delay for each airline by month

sns.heatmap(data=corr, annot=True)
# set zeros to Nan, and get correlation coefficients.

mydata = train[my_vars].replace(0, np.nan)
# check how many missings we have per feature

missings = mydata.isnull().sum().to_frame()



# give column the name nMissings

missings.columns = ['nMissings']



# make new columns that gives information about missing percentage

missings['percMissing'] = missings['nMissings']/1460



# sort values by nMissings values

ordered_missings = missings.sort_values(by = ['nMissings'], ascending=False)

ordered_missings
corr = mydata.corr()



# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Correlation between each feature related to size and SalePrice")



# Heatmap showing correlation coefficients

sns.heatmap(data=corr, annot=True)
my_vars = ['1stFlrSF', 'LotArea', 'GrLivArea', 'SalePrice']

mydata = train[my_vars].replace(0, np.nan)

for i in my_vars:

    print(i)

    print("Skewness before log transformation: %f" % mydata[i].skew())

    print("Kurtosis before log transformation: %f" % mydata[i].kurt())
for i in my_vars:

    print(i)

    print("Skewness after log transformation: %f" % np.log(mydata[i]).skew())

    print("Kurtosis after log transformation: %f" % np.log(mydata[i]).kurt())
corr =np.log(mydata[my_vars]).corr()



# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Correlation between each feature related to size and SalePrice")



# Heatmap showing correlation coefficients

sns.heatmap(data=corr, annot=True)
train.head()
train.MSZoning.unique()



# break data into different parts

rl = train[train.MSZoning == 'RL']

rm = train[train.MSZoning == 'RM']

c = train[train.MSZoning == 'C (all)']

fv = train[train.MSZoning == 'FV']

rh = train[train.MSZoning == 'RH']



# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Histograms for each species

sns.distplot(a = np.log(rl['SalePrice']), label="Residential Low Density", kde=False)

sns.distplot(a = np.log(rm['SalePrice']), label="Residential Medium Density", kde=False)

sns.distplot(a = np.log(c['SalePrice']), label="Commercial", kde=False)

sns.distplot(a = np.log(fv['SalePrice']), label="Floating Village Residential", kde=False)

sns.distplot(a = np.log(rh['SalePrice']), label="Residential High Density", kde=False)



# Add title

plt.title("Histogram of SalePrice, by MSzoning")



# Force legend to appear

plt.legend()