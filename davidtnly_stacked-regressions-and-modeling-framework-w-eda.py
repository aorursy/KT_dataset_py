from IPython.display import Image

url = 'https://gisgeography.com/wp-content/uploads/2014/07/rmse-formula1-300x96.png'

# Image(url, width=300, height=350)
import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Toolbox 101

import pandas_profiling

import numpy as np

import random as rand

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from scipy import stats



# Evaluation

from sklearn.metrics import mean_squared_error #RMSE

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")



# Preset data display

pd.options.display.max_seq_items = 5000

pd.options.display.max_rows = 5000



# Set palette

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set_palette(flatui)

sns.palplot(sns.color_palette(flatui))

#34495e
# Import data

train = pd.read_csv('09-house-train.csv')

test = pd.read_csv('09-house-test.csv')
# Check data

train.head()
test.head()
# Drop

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
# Check

train.iloc[0:5,:3]
'''



Some functions to start off with:



train.sample()                                           

train.describe()

    train.describe(include=['O'])

    train.describe(include='all')

train.head()

train.tail()

train.value_counts().sum()

train.isnull().sum()

train.count()

train.fillna()

    train.fillna(train[col].mode(), inplace=True)

train.mean()

train.median()

train.mode()

train.shape

train.info()



'''
# Get data shape, info, columns, & dimensions

print ("*"*40)

print('********** train shape: ' + str(train.shape) + '*'*10)

print (train.info())

print ("*"*40)

print('********** test shape: ' + str(test.shape) + '*'*10)
# Get null pct and counts

null_cols = pd.DataFrame(train.isnull().sum().sort_values(ascending=False), columns=['Null Data Count'])

null_cols_pct = pd.DataFrame(round(train.isnull().sum().sort_values(ascending=False)/len(train),2)*100, columns=['Null Data Pct'])



# Combine horizontally (axis=1) into a dataframe with column names (keys=[]) then to a data frame

null_cols_df = pd.DataFrame(pd.concat([null_cols, null_cols_pct], axis=1))



all_nulls = null_cols_df[null_cols_df['Null Data Pct']>0]



print('There are', len(all_nulls), 'columns with missing values.')

all_nulls
# Create figure space

plt.figure(figsize=(12,8))



# Create plot

sns.barplot(x=all_nulls.index,

            y='Null Data Pct',

            data=all_nulls)



# Set plot features

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of Missing Values', fontsize=15)

plt.title('Percent of Missing Data by Feature', fontsize=15)
# Create a dataframe to store the values

saleprice_df = pd.concat([train.SalePrice, np.log(train.SalePrice+1).rename('LogSalePrice')],

                          axis=1, names=['SalePrice', 'LogSalePrice'])

saleprice_df.head()
# Drop

train = train.drop(train[(train.SalePrice>450000)].index)



# Drop column example

# .drop('Cabin', axis=1, inplace=True)
sns.set_style("white")

sns.set_color_codes(palette='deep')



# Create figure space

fig, ax = plt.subplots(figsize=(18,5), ncols=2, nrows=1)



# Create a distribution plot

ax1 = sns.distplot(saleprice_df.SalePrice, kde=False, fit=norm, ax=ax[0])

ax2 = sns.distplot(saleprice_df.LogSalePrice, kde=False, fit=norm, ax=ax[1])



# Set plot features

ax1.set_title('SalePrice Distribution')

ax2.set_title('LogSalePrice Distribution')
# Create figure space

plt.figure(figsize=(10,5))



# Create plot

sns.distplot(train['SalePrice'] , fit=norm)



# Get the fitted parameters (feels off without the params somewhere visible)

mu, sigma = norm.fit(train['SalePrice'])

print( '\n mu = {:.0f} and sigma = {:.0f}\n'.format(mu, sigma))



# Plot distribution

plt.legend(['Norm Dist. ($\mu=$ {:.0f} and $\sigma=$ {:.0f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
# Skewness and kurtosis

print('Skewness: %f' % train['SalePrice'].skew())

print('Kurtosis: %f' % train['SalePrice'].kurt())
url = 'https://i.imgur.com/yIqX5W5.jpg'

# Image(url, width=500, height=500)
# Return the natural logarithm of one plus the input array, element-wise.

train['LogSalePrice'] = np.log1p(train.SalePrice)
stats.probplot(train['SalePrice'], plot=plt)

plt.show();
stats.probplot(np.log(train.SalePrice), plot=plt)

plt.show();
# Count data types in the train dataset

train.dtypes.value_counts()
# Find numeric features

num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_data = train.select_dtypes(include=num_dtypes)



# Find all other features

col_data = train.select_dtypes(include=['object'])
print(num_data.head(2))

print(col_data.head(2))
# Correlation to LogSalePrice feature

pd.DataFrame(abs(num_data.corr()['LogSalePrice']).sort_values(ascending=False))
# Top Features

r_squared = num_data.corr()**2

r_squared.LogSalePrice.sort_values(ascending=False)
plt.figure(figsize=(15, 10))

flights = sns.load_dataset("flights")

flights = flights.pivot("month", "year", "passengers")

ax = sns.heatmap(flights, annot=True, fmt="d")



for text in ax.texts:

    text.set_size(14)

    if text.get_text() == '118':

        text.set_size(18)

        text.set_weight('bold')

        text.set_style('italic')
# Create a figure space

plt.subplots(figsize=(12,9))



# Create matrix

corr_plot = sns.heatmap(num_data.corr(),

#                         annot=True,

                          cmap='viridis', # YlGnBu, RdBu_r

                          linewidths=0.20,

                          linecolor='white',

                          vmax=1,

                          square=True,

                          fmt='.1g',

                          annot_kws={"size": 12})

corr_plot
sns.set_style('whitegrid')

ax = train.hist(bins=20, figsize=(15,15), grid=False)

plt.show();
# Some numerical features

plot_list = ['SalePrice', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'TotalBsmtSF', 'GrLivArea']



# Plot pairplot

sns.pairplot(train[plot_list])
### Train vs Test Distribution
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))



ax1.set_title('SalePrice')

sns.kdeplot(train['SalePrice'], ax=ax1, label="train")

sns.kdeplot(test['SalePrice'], ax=ax1, label="test")



plt.show()

# Create figure space

fig, ax = plt.subplots(figsize=(10,8))



# Create boxplot

ax = sns.boxplot(x=train.OverallQual,

                 y=train.SalePrice,

                 data=train)



# Set plot features

ax.set_title('Overall Quality vs. SalePrice', fontsize=15)
# Create figure space

fig, ax = plt.subplots(figsize=(20,10))



# Create boxplot

ax = sns.boxplot(x=train.YearBuilt,

                 y=train.SalePrice,

                 data=train)



# Set plot features

ax.set_title('Year Built Sale Price Distribution', fontsize=20)

ax.set_xlabel('Year Built', fontsize=15)

ax.set_ylabel('Sale Price', fontsize=15)

ax.set(xticklabels=[])
# Remove white grid

sns.set_style('whitegrid')



# Set figure space

fig = plt.figure(figsize=(8, 6))



# Create scatterplot with a loess line

import statsmodels

sns.regplot(x='YearBuilt', y='SalePrice', data=train, lowess=True,

            color='#34495e', scatter=True,

            line_kws={'color': 'red'},

            scatter_kws={'alpha': 0.50})



# sns.scatterplot(x='YearBuilt',

#                 y='SalePrice',

#                 data=train)



# Set plot features

plt.title('Year Built vs. SalePrice', fontsize=15)

plt.xlabel('Year Built', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

plt.ylim(0, 800000)

plt.xlim(1860, 2020)
# Define the function

def hist_qq_plot(var):

    # Get normal distribution 

    sns.distplot(var, fit=norm)

    fig = plt.figure()

    qq = stats.probplot(var, plot=plt,)

#     qq.get_lines()[0].set_markerfacecolor('#34495e')
hist_qq_plot(train.YearBuilt)
url = 'https://serving.photos.photobox.com/3262242285a81460d733332055b13e4ce0aaffc306c6d4165167c76251183608696600e0.jpg'

Image(url, width=500, height=500)
# Create figure space

fig, ax = plt.subplots(figsize=(20,10))



# Create boxplot

ax = sns.boxplot(x=train.MoSold,

                 y=train.SalePrice,

                 data=train)



# Set plot features

ax.set_title('Month Sold Price Distribution', fontsize=20)

ax.set_xlabel('Month', fontsize=15)

ax.set_ylabel('Sale Price', fontsize=15)
# Set figure space

fig, ax = plt.subplots(figsize=(20,10), ncols=2)



# Create countplot()

ax1 = sns.countplot(x='MoSold',

                   data=train,

                   linewidth=2,

                   ax=ax[0]

                   )



ax2 = sns.distplot(train.MoSold, rug=True, ax=ax[1]) 



# # Plot the distribution with a histogram and maximum likelihood gaussian distribution fit

# ax2 = sns.distplot(train.MoSold, rug=True, fit=norm, kde=False, ax=ax[1]) 



# Set plot features

ax1.set_title('Homes Sold by Month', fontsize=15)

ax1.set_xlabel('Month', fontsize=15)

ax1.set_ylabel('Units', fontsize=15)



ax2.set_title('Homes Sold by Month Univariate Distribution', fontsize=15)

ax2.set_xlabel('Month', fontsize=15)

ax2.set_ylabel('Density', fontsize=15)
# Create figure space

fig, ax = plt.subplots(figsize=(20,10))



# Create boxplot

ax = sns.boxplot(x=train.YrSold,

                 y=train.SalePrice,

                 data=train)



# Set plot features

ax.set_title('Year Sold Price Distribution', fontsize=20)

ax.set_xlabel('Year', fontsize=15)

ax.set_ylabel('Sale Price', fontsize=15)
# Create figure space

fig, ax = plt.subplots(figsize=(20, 10), ncols=2)



# Create scatterplot

ax1 = sns.scatterplot(x='YearBuilt',

                      y='SalePrice',

                      data=train,

                      ax=ax[0])



ax2 = sns.scatterplot(x='YearBuilt',

                      y='YrSold',

                      data=train,

                      ax=ax[1])



# Set plot features

ax1.set_title('Year Built vs. Sales Price', fontsize=15)

ax2.set_title('Year Built vs. Year Sold', fontsize=15)
# Remove outliers after viewing plots from below

# Drop

train = train.drop(train[(train['1stFlrSF']>2500)].index)
# Set figure space

fig, ax = plt.subplots(figsize=(16,10), ncols=2)



# Create scatterplot with a loess line

ax1= sns.regplot(x='1stFlrSF', y='SalePrice', data=train,

                 lowess=True, color='#34495e', scatter=True,

                 line_kws={'color': 'red'}, ax=ax[0])



# There looks to be a bit of outliers that skews the visual, let's plot without it

ax2 = sns.regplot(x='1stFlrSF', y='SalePrice',

                  data=train[(train['1stFlrSF']<2500)], lowess=True,

                  color='#34495e', scatter=True, line_kws={'color': 'red'},

                  ax=ax[1])



# Set plot features

ax1.set_title('1st Floor SqFt vs. SalePrice', fontsize=15)

ax1.set_xlabel('SqFt', fontsize=15)

ax1.set_ylabel('Sale Price', fontsize=15)



ax2.set_title('1st Floor SqFt vs. SalePrice', fontsize=15)

ax2.set_xlabel('SqFt', fontsize=15)

ax2.set_ylabel('Sale Price', fontsize=15)
hist_qq_plot((train['1stFlrSF']))
hist_qq_plot(np.log(train['1stFlrSF']))
# Remove outliers after viewing plots from below

# Drop

train = train.drop(train[(train['GrLivArea']>4000)].index)
# Set figure space

plt.figure(figsize=(8, 6))



# Create scatterplot with loess

sns.regplot(x='GrLivArea', y='SalePrice', data=train,

            lowess=True, scatter=True, color='#34495e',

            line_kws={'color': 'red'},

            scatter_kws={'alpha': 0.50})



# Plot diagonal line

plt.plot([0, 6000], [0, 800000], 'darkorange', lw=2, linestyle='--')



# Set plot features

plt.title('GrLivArea vs. SalePrice', fontsize=15)

plt.xlabel('Area', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

plt.xlim(0, 4000)

plt.ylim(0, 500000)
hist_qq_plot((train['GrLivArea']))
hist_qq_plot(np.log(train['GrLivArea']))
# Remove outliers after viewing plots from below

# Drop

train = train.drop(train[(train['TotalBsmtSF']>3000)].index)
# Set figure space

plt.figure(figsize=(8, 6))



# Create scatterplot with loess

sns.regplot(x='TotalBsmtSF', y='SalePrice', data=train[train.TotalBsmtSF<6000],

            lowess=True, scatter=True, color='#34495e',

            line_kws={'color': 'red'},

            scatter_kws={'alpha': 0.50})



# Plot diagonal line

# plt.plot([0, 6000], [0, 800000], 'darkorange', lw=2, linestyle='--')



# Set plot features

plt.title('Total Basement SqFt vs. SalePrice', fontsize=15)

plt.xlabel('SqFt', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

# plt.xlim(0, 6000)

# plt.ylim(0, 800000)
hist_qq_plot((train['TotalBsmtSF']))
hist_qq_plot(np.log(train[train['TotalBsmtSF']>0].TotalBsmtSF))
def pct_bar_labels():

    '''

    Function used to label the relative frequency on top of each bars

    '''

    # Set font size

    fs=15

    

    # Set plot label and ticks

    plt.ylabel('Relative Frequency (%)', fontsize=fs)

    plt.xticks(rotation=0, fontsize=fs)

    plt.yticks([])

    

    # Set individual bar labels in proportional scale

    for x in ax1.patches:

        ax1.annotate(str(x.get_height()) + '%', 

        (x.get_x() + x.get_width()/2., x.get_height()), ha='center', va='center', xytext=(0, 7), 

        textcoords='offset points', fontsize=fs, color='black')



def freq_table(var):

    '''

    Define plot global variables

    Create a function that will populate a frequency table (%)

    Get counts per feature then get the percentage over the total counts

    '''

    global ax, ax1

    

    # Get Values and pct and combine it into a dataframe

    count_freq = var.value_counts()

    pct_freq = round(var.value_counts(normalize=True)*100, 2)

    

    # Create a dataframe

    df = pd.DataFrame({'Count': count_freq, 'Percentage': pct_freq})

    

    # Print variable name

    print('Frequency of', var.name, ':')

    display(df)

    

    # Create plot

    ax1 = pct_freq.plot.bar(title='Percentage of {}'.format(var.name), figsize=(12,8))

    ax1.title.set_size(15)

    pct_bar_labels()

    plt.show()
freq_table(train.BsmtQual)
# Set figure size

# fig, ax = plt.subplots(figsize=(16,10), ncols=2)



# Create plots: 1 for bsmtqual/totalbsmtsf & 1 for bsmtqual/saleprice

ax1 = sns.catplot(x='BsmtQual',

                  y='TotalBsmtSF',

                  data=train[train.TotalBsmtSF<6000],

                  kind='box',

                  order=['Fa', 'TA', 'Gd', 'Ex']

                  )



ax2 = sns.catplot(x='BsmtQual',

                  y='SalePrice',

                  data=train,

                  kind='box',

                  order=['Fa', 'TA', 'Gd', 'Ex']

                  )



# Set plot features

ax1.fig.suptitle('BsmtQual vs. TotalBsmtSF', fontsize=12)

# ax1.subplots_adjust(top=0.80) # for facet only

ax2.fig.suptitle('BsmtQual vs. SalePrice', fontsize=12)
# Create the figure space

fig, ax = plt.subplots(figsize=(12, 8))



# Create boxplots

ax = sns.boxplot(x='FireplaceQu',

                 y='SalePrice',

                 data=train,

                 order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])



# Set plot features

ax.set_title('Fireplace Quality Distribution', fontsize=15)

ax.set_xlabel('Fireplace Quality', fontsize=12)

ax.set_ylabel('Sale Price', fontsize=12)
freq_table(train.FireplaceQu)
# Create the figure space

fig, ax = plt.subplots(figsize=(12, 8))



# Create boxplots

ax = sns.boxplot(x='MSZoning',

                 y='SalePrice',

                 data=train)

#                  order=['Po', 'Fa', 'TA', 'Gd', 'Ex'])



# Set plot features

ax.set_title('MSZoning Distribution', fontsize=15)

ax.set_xlabel('MSZoning', fontsize=12)

ax.set_ylabel('Sale Price', fontsize=12)
# Set figure space

plt.figure(figsize=(8, 6))



# Create scatterplot with loess line

sns.regplot(x='LotArea',

            y='SalePrice',

            data=train,

            lowess=True,

            scatter=True,

            color='#34495e',

            line_kws={'color': 'red'},

            scatter_kws={'alpha': 0.50})



# Set plot features

plt.title('Lot Area vs. Sale Price', fontsize=15)

plt.xlabel('Lot Area', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

# Drop outliers

train = train.drop(train[(train.LotArea>30000)].index)
# Set figure space

plt.figure(figsize=(8, 6))



# Create scatterplot with loess line

sns.regplot(x='LotArea',

            y='SalePrice',

            data=train,

            lowess=True,

            scatter=True,

            color='#34495e',

            line_kws={'color': 'red'},

            scatter_kws={'alpha': 0.50})



# Set plot features

plt.title('Lot Area vs. Sale Price', fontsize=15)

plt.xlabel('Lot Area', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)
# Looks more normal removing the additional points

hist_qq_plot(train[train.LotArea<30000].LotArea)
# See if I really want to remove them

train[train.LotArea>25000]
train['YrRemodel_Diff'] = train['YearRemodAdd'] - train['YearBuilt']

test['YrRemodel_Diff'] = test['YearRemodAdd'] - test['YearBuilt']
# Create figure space

fig, ax = plt.subplots(figsize=(20,10))



# Create boxplot

ax = sns.boxplot(x='YrRemodel_Diff',

                 y='SalePrice',

                 data=train)



# Set plot features

plt.xticks(rotation='45')

ax.set_title('Year Remodel Difference', fontsize=15)

ax.set_xlabel('Year Difference', fontsize=12)

ax.set_ylabel('Sale Price', fontsize=12)

ax.set(xticklabels=[])
hist_qq_plot(train.YrRemodel_Diff)
# Let's plot with the zeros

hist_qq_plot(train[train.YrRemodel_Diff>1].YrRemodel_Diff)
# Define function

def create_remodel_diff_bin(var):

    grp=''

    if var==0:

        grp='Never Remodeled'

    elif var>=1 and var <11:

        grp='1-10'

    elif var>=11 and var <21:

        grp='11-20'

    elif var>=21 and var <31:

        grp='21-30'

    elif var>=31 and var <41:

        grp='31-40'

    elif var>=41 and var <51:

        grp='41-50'

    elif var>=51 and var <61:

        grp='51-60'

    elif var>=61 and var <71:

        grp='61-70'

    else:

        grp='71+'

    return grp
train['Yr_Remodel_Group'] = train.YrRemodel_Diff.map(create_remodel_diff_bin)

test['Yr_Remodel_Group'] = test.YrRemodel_Diff.map(create_remodel_diff_bin)

train.head()
# Create figure space

fig, ax = plt.subplots(figsize=(15,10))



# Create boxplot

ax = sns.boxplot(x='Yr_Remodel_Group',

                 y='SalePrice',

                 data=train)



# Set plot features

plt.xticks(rotation='45')

ax.set_title('Year Remodel Group', fontsize=15)

ax.set_xlabel('Year Group', fontsize=12)

ax.set_ylabel('Sale Price', fontsize=12)
# Create figure space

fig, ax = plt.subplots(figsize=(12,8))



# Create jitter

ax = sns.stripplot(x='Yr_Remodel_Group',

                   y='SalePrice',

                   data=train,

                   jitter=True,

                   alpha=0.50)



# Set plot features

plt.xticks(rotation='45')

ax.set_title('Year Remodel Distribution', fontsize=15)

ax.set_xlabel('Year Group', fontsize=12)

ax.set_ylabel('Sale Price', fontsize=12)
train.drop(['YrRemodel_Diff'], axis=1, inplace=True)

test.drop(['YrRemodel_Diff'], axis=1, inplace=True)
# # Create facet grid

# ax = sns.FacetGrid(train,

#                    col='SaleType',

#                    row='Yr_Remodel_Group',

#                    margin_titles=True,

#                    hue='SaleType')



# # Create plot using map()

# ax = ax.map(plt.hist, 'SalePrice', edgecolor='w')



# Create figure space

fig, ax = plt.subplots(figsize=(15,6), ncols=2, nrows=1)



# Create plot

ax1 = sns.boxplot(x='SaleType',

                 y='SalePrice',

                 data=train,

                 ax=ax[0])



ax2 = sns.stripplot(x='SaleType',

                   y='SalePrice',

                   data=train,

                   jitter=True,

                   alpha=0.50,

                   edgecolor='w',

                   ax=ax[1])



# Set plot features

ax1.set_title('Sale Type Distribution by BoxPlot', fontsize=15)

ax2.set_title('Sale Type Distribution by Points', fontsize=15)
# Create figure space

fig, ax = plt.subplots(figsize=(15,6), ncols=2, nrows=1)



# Create plot

ax1 = sns.scatterplot(x='LotFrontage',

                      y='SalePrice',

                      data=train,

                      ax=ax[0])



ax2 = sns.stripplot(x='LotFrontage',

                   y='SalePrice',

                   data=train,

                   jitter=True,

                   alpha=0.50,

                   edgecolor='w',

                   ax=ax[1])



# Set plot features

ax1.set_title('Lot Area Distribution', fontsize=15)

ax2.set_title('Lot Area Distribution', fontsize=15)

ax2.set(xticklabels=[])
# Features that pertains to area

ax = sns.pairplot(train[['SalePrice', 'LotArea', '1stFlrSF', 'LotFrontage']])

ax.set(xticklabels=[])

plt.show()
train['HouseAge'] = train.YrSold.max() - train.YearBuilt

test['HouseAge'] = test.YrSold.max() - test.YearBuilt
# Set colors

# colors = {1: 'lightblue', 0: 'gray'}



# Create a figure

plt.figure(figsize=(20,15))



# Create facet grid

ax = sns.FacetGrid(train,

                   col='MoSold',

                   hue='CentralAir',

                   margin_titles=True)

#                    palette=colors)



# Create scatter

ax.map(plt.scatter, 'HouseAge', 'SalePrice', edgecolor='w') # , s=100)



# Add a legend

ax.add_legend()



# Set plot features

ax.fig.suptitle('Sale Price by Month Sold, Central Air Conditioning & House Age', size=15)

# plt.subplots_adjust(top=0.85)

plt.show()
# Calling corr_plot wasn't showing the plot so copy and pasting it again



# Create a figure space

plt.subplots(figsize=(12,9))



# Create matrix

corr_plot = sns.heatmap(num_data.corr(),

#                         annot=True,

                          cmap='viridis', # YlGnBu, RdBu_r

                          linewidths=0.20,

                          linecolor='white',

                          vmax=1,

                          square=True,

                          fmt='.1g',

                          annot_kws={"size": 12})

corr_plot
# Set figure space

plt.figure(figsize=(15,10))



# Set k (number of variables for the heatmap)

k = 15



# Create correlation using corr.nlargest()

top_corr = num_data.corr().nlargest(k, 'SalePrice')['SalePrice'].index



# Get correlation coefficient

cm = np.corrcoef(train[top_corr].values.T)



# Set plot scale

sns.set(font_scale=1.25)



# Create heatmap

top_corr_plot = sns.heatmap(cm, cbar=True, annot=True, square=True,

                            fmt='.2f', annot_kws={'size': 12}, yticklabels=top_corr.values,

                            xticklabels=top_corr.values)

plt.show()
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
# Plot 

hist_qq_plot((train['TotalSF']))
# Check the outliers to make sure it was dropped already

train[train.TotalSF>6000]
pandas_profiling.ProfileReport(train)



# # Output to a html file if needed

# profile = pandas_profiling.ProfileReport(train)

# profile.to_file(outputfile='House regression data profiling.html')
# Find numeric features

num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_data = train.select_dtypes(include=num_dtypes)



# Find all other features

col_data = train.select_dtypes(include=['object'])
# Copy data to make sure we don't mess up and if we do just rerun

model_train = train.copy()



# Replace categorical features with none

fill_col_columns = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition', 'Yr_Remodel_Group']



# Replace categorical features with mode

fill_mode_columns = ['SaleType', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical', 'MSZoning']



# None values

for i in fill_col_columns:

    model_train[i] = model_train[i].fillna('None')



# Mode values

for i in fill_mode_columns:

    model_train[i] = model_train[i].fillna(model_train[i].mode())



# Specific values

model_train['Functional'] = model_train['Functional'].fillna('Typ')

model_train['Electrical'] = model_train['Electrical'].fillna('SBrkr')
fill_num_columns = ['MSSubClass','LotArea', 'OverallQual', 'OverallCond',

       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',

       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold', 'YrSold', 'SalePrice', 'LogSalePrice', 'TotalSF', 'HouseAge']



# Zero values

for i in fill_num_columns:

    model_train[i] = model_train[i].fillna(0)

    

# Specific values

'''

Look up differences between these three

transform() is an operation used in conjunction with groupby (which is one of the most useful operations in pandas)

vs. map()

vs apply()

'''

model_train['LotFrontage'] = model_train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
model_train.head()
# Check for nulls

# model_train.isnull().sum().sort_values(ascending=False)
# train['NewHouse'] = [1 if train['YearBuilt'] == train['YrSold'] else 0]



# NewHouse = []

# for i in train:

#     if train.loc[i, 'YearBuilt'] == train.loc[i, 'YrSold']:

#         NewHouse.append('1')

#     else:

#         NewHouse.append('0')



# # Define a function to get NewHouse

# def new_house(df):

#     text=''

#     if df == train['YrSold']:

#         text='Yes'

#     else:

#         text='No'

#     return text



# train['NewHouse'] = train.YearBuilt.map(new_house)
# Create new features based on conditions

model_train['NewHouse'] = np.where(model_train['YearBuilt']==model_train['YrSold'], 1, 0)

model_train['TotalLotArea'] = model_train.LotFrontage + model_train.LotArea

model_train['OverallQualityCondition'] = model_train.OverallCond + model_train.OverallQual



# Basically flags if a house has a certain feature

model_train['HasWoodDeck'] = (model_train['WoodDeckSF'] == 0) * 1 # found this from another kernel, interesting way to write method

model_train['HasOpenPorch'] = (model_train['OpenPorchSF'] == 0) * 1

model_train['HasEnclosedPorch'] = (model_train['EnclosedPorch'] == 0) * 1

model_train['Has3SsnPorch'] = (model_train['3SsnPorch'] == 0) * 1

model_train['HasScreenPorch'] = (model_train['ScreenPorch'] == 0) * 1

model_train['HasPool'] = model_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

model_train['Has2ndfloor'] = model_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

model_train['HasGarage'] = model_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

model_train['HasBsmt'] = model_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

model_train['HasFireplace'] = model_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



# Total Bathrooms - a big feature, definitely something I would look at 

model_train['TotalBathrooms'] = (model_train['FullBath'] + (0.5*model_train['HalfBath']) +

                                 model_train['BsmtFullBath'] + (0.5*model_train['BsmtHalfBath']))



##########################################################

# Create new features based on conditions

test['NewHouse'] = np.where(test['YearBuilt']==test['YrSold'], 1, 0)

test['TotalLotArea'] = test.LotFrontage + test.LotArea

test['OverallQualityCondition'] = test.OverallCond + test.OverallQual



# Basically flags if a house has a certain feature

test['HasWoodDeck'] = (test['WoodDeckSF'] == 0) * 1

test['HasOpenPorch'] = (test['OpenPorchSF'] == 0) * 1

test['HasEnclosedPorch'] = (test['EnclosedPorch'] == 0) * 1

test['Has3SsnPorch'] = (test['3SsnPorch'] == 0) * 1

test['HasScreenPorch'] = (test['ScreenPorch'] == 0) * 1

test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test['Has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test['HasGarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['HasFireplace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



# Total Bathrooms - a big feature, definitely something I would look at 

test['TotalBathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) +

                          test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))

# Check new feature to see if this makes sense

model_train.groupby('NewHouse').SalePrice.mean()
## Shorter version

# numeric_features = model_train.dtypes[model_train.dtypes != "object"].index

# Check the skew of all numerical features

# skew_features = model_train[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

# print("\nSkew in numerical features: \n")

# skews = pd.DataFrame({'Skew' :skew_features})

# skews.head()



# For loop version

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



numeric_list = []

for i in model_train.columns:

    if model_train[i].dtype in numeric_dtypes: 

        numeric_list.append(i)



# Dataframe has no attribute map() so we have to use the apply() fuction

skew_features = model_train[numeric_list].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'Skew':skew_features})

skews.head(10)
from scipy.special import boxcox1p # Compute the Box-Cox transformation of 1 + x

from scipy.stats import boxcox_normmax # Compute optimal Box-Cox transform parameter for input data



# Get high skews

high_skew = skew_features[skew_features>0.50] # mid

skew_index = high_skew.index 

print('There are {} skewed features.'.format(high_skew.shape[0]))



# Loop through the index and transform

for i in skew_index:

    model_train[i] = boxcox1p(model_train[i], boxcox_normmax(model_train[i]+1))



# Get new transformed features

new_skew_features = model_train[numeric_list].apply(lambda x: skew(x)).sort_values(ascending=False)

new_skews_df = pd.DataFrame({'Skew': new_skew_features})

new_skews_df.head(15)
# Check corr

pd.DataFrame(abs(train.corr()['SalePrice']).sort_values(ascending=False))
model_train.columns
# Drop any feature that we have not yet

model_train.drop(['SalePrice', 'YearRemodAdd'], axis=1, inplace=True)

test.drop(['YearRemodAdd'], axis=1, inplace=True)
# Get data shape, info, columns, & dimensions

print ("*"*40)

print('********** train shape: ' + str(model_train.shape) + '*'*10)

print ("*"*40)

print('********** test shape: ' + str(test.shape) + '*'*10)
# Convert to str

train['MSSubClass'] = train['MSSubClass'].apply(str)

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)



# Wonder if I should update this here as well - i'll leave it for now

test['MSSubClass'] = test['MSSubClass'].apply(str)

test['YrSold'] = test['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
# Create a list of features for specific dummy variables

dummy_list = []

model_train = pd.get_dummies(model_train).reset_index(drop=True) 

                           # drop_first=False

                            # Whether to get k-1 dummies out of k categorical levels by removing the first level

    

# Other parameters if there is a certain list: columns=dummy_list
model_train_dupe = model_train.copy()

model_train.shape
# Remove any duplicated column names

model_train = model_train.loc[:, ~model_train.columns.duplicated()]
# ### Adding in top features from xgb and rf (this step is found after feature importance)

# top_features_list = 

# ['1stFlrSF',

#  '2ndFlrSF',

#  'Alley_Grvl',

#  'BedroomAbvGr',

#  'BldgType_1Fam',

#  'BsmtCond_Fa',

#  'BsmtFinSF1',

#  'BsmtFinType1_Unf',

#  'BsmtQual_Ex',

#  'BsmtQual_Gd',

#  'BsmtUnfSF',

#  'CentralAir_N',

#  'CentralAir_Y',

#  'EnclosedPorch',

#  'ExterQual_Fa',

#  'Exterior1st_BrkComm',

#  'FireplaceQu_None',

#  'Fireplaces',

#  'Foundation_BrkTil',

#  'Foundation_PConc',

#  'FullBath',

#  'Functional_Maj1',

#  'Functional_Sev',

#  'Functional_Typ',

#  'GarageArea',

#  'GarageCars',

#  'GarageCond_TA',

#  'GarageFinish_Fin',

#  'GarageFinish_Unf',

#  'GarageType_Attchd',

#  'GarageType_Detchd',

#  'GarageYrBlt',

#  'GrLivArea',

#  'HasFireplace',

#  'HeatingQC_Ex',

#  'HeatingQC_Fa',

#  'Heating_GasA',

#  'Heating_Grav',

#  'HouseAge',

#  'KitchenAbvGr',

#  'KitchenQual_Ex',

#  'KitchenQual_Fa',

#  'KitchenQual_Gd',

#  'KitchenQual_TA',

#  'LotArea',

#  'LotFrontage',

#  'LotShape_Reg',

#  'MSSubClass',

#  'MSZoning_C (all)',

#  'MSZoning_RL',

#  'MSZoning_RM',

#  'MasVnrArea',

#  'MoSold',

#  'Neighborhood_Crawfor',

#  'Neighborhood_OldTown',

#  'OpenPorchSF',

#  'OverallCond',

#  'OverallQual',

#  'OverallQualityCondition',

#  'SaleCondition_Abnorml',

#  'SaleCondition_Normal',

#  'SaleType_New',

#  'TotRmsAbvGrd',

#  'TotalBathrooms',

#  'TotalBsmtSF',

#  'TotalLotArea',

#  'TotalSF',

#  'WoodDeckSF',

#  'YearBuilt',

#  'YrSold',

#  'Yr_Remodel_Group_31-40',

#  'LogSalePrice']
# # Top features

# model_train = model_train[top_features_list]
# Save the actual test set to another variable

test_dupe = test.copy()
# Split the model data to train and test

train = model_train

print('Train data shape: ' + str(train.shape))





# train = model_train[1:891]

# test = model_train[892:model_train.shape[0]]

# print('Test data shape: ' + str(test.shape))
# Split

y = train['LogSalePrice']

X = train.drop(['LogSalePrice'], axis=1)
# Import split module

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)



# Check

X_train.head(3)
X_test.shape
y_test.shape
from mlxtend.regressor import StackingCVRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.ensemble import AdaBoostRegressor

# from sklearn.ensemble import BaggingRegressor

# from sklearn.ensemble import ExtraTreesRegressor



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LassoCV

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import ElasticNetCV

from sklearn.svm import SVR # SVC = classification



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor, plot_importance 
# Set seed

seed = 150



# Instantiate baseline model

lm = LinearRegression()



# Instantiate additional base models

rf = RandomForestRegressor(random_state=seed)

gbr = GradientBoostingRegressor(random_state=seed)

# abr =AdaBoostRegressor(random_state=seed)

# br = BaggingRegressor(random_state=seed)

# etr = ExtraTreesRegressor(random_state=seed)

lasso_cv = LassoCV()

ridge_cv = RidgeCV()

glmnet_cv = ElasticNetCV(random_state=seed)

svr = SVR()

lgbmr = LGBMRegressor(random_state=seed)

xgbr = XGBRegressor(random_state=seed)
# Set CV method

cv = KFold(n_splits=10, random_state=150, shuffle=True)
# Define the function for mean_squared_error() evaluation for the blended model

def base_rmse(y, y_pred):

    ''' 

    Return the sqrt of mean the mean squared error between the two values

    '''

    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return rmse



# Define the function for a cross-validation evaluation for meta-learners

def cv_rmse(model, X=X_train, y=y_train):

    '''

    Return the sqrt of mean the cross-validated mean squared error between the two values

    Replace the default parameter arguments X & y when we use testing data

    '''

    cv_rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))

    return cv_rmse
# Linear Regression

lm.fit(X_train, y_train)

lm_y_pred = lm.predict(X_test)

lm_rmse = np.sqrt(mean_squared_error(y_test, lm_y_pred))

print('Linear Regression Base Model Score: {:.2f}'.format(lm_rmse))
# Linear Regression

starttime = dt.datetime.now()

lm = make_pipeline(RobustScaler(), lm).fit(X=X_train, y=y_train)

lm_score = cv_rmse(lm).mean()

endtime = dt.datetime.now()

lm_build_time = endtime-starttime



# Random Forest

starttime = dt.datetime.now()

rf = RandomForestRegressor(n_estimators=1000,

#                            max_depth=10,

#                            min_samples_split=5,

#                            min_samples_leaf=5,

#                            max_features=None,

#                            oob_score=True

                          )

rf_score = cv_rmse(rf).mean()

endtime = dt.datetime.now()

rf_build_time = endtime-starttime



# Gradient Boosting

starttime = dt.datetime.now()

gbr = GradientBoostingRegressor(n_estimators=1000,

                                learning_rate=0.01,

#                                 max_depth=5,

#                                 max_features='sqrt',

#                                 min_samples_leaf=15,

#                                 min_samples_split=10, 

#                                 loss='huber',

                                random_state=100

                               )

gbr_score = cv_rmse(gbr).mean()

endtime = dt.datetime.now()

gbr_build_time = endtime-starttime



# Lasso CV

starttime = dt.datetime.now()

lasso_cv = make_pipeline(RobustScaler(), LassoCV(cv=cv, random_state=seed))

lasso_cv_score = cv_rmse(lasso_cv).mean()

endtime = dt.datetime.now()

lasso_cv_build_time = endtime-starttime



# Ridge CV

starttime = dt.datetime.now()

ridge_cv = make_pipeline(RobustScaler(), RidgeCV(cv=cv))

ridge_cv_score = cv_rmse(ridge_cv).mean()

endtime = dt.datetime.now()

ridge_cv_build_time = endtime-starttime



# GLMNET CV

starttime = dt.datetime.now()

glmnet_cv = make_pipeline(RobustScaler(), ElasticNetCV(cv=cv))

glmnet_cv_score = cv_rmse(glmnet_cv).mean()

endtime = dt.datetime.now()

glmnet_cv_build_time = endtime-starttime



# Support Vector

starttime = dt.datetime.now()

svr = make_pipeline(RobustScaler(), SVR())

svr_score = cv_rmse(svr).mean()

endtime = dt.datetime.now()

svr_build_time = endtime-starttime



# Light Gradient Boosting

starttime = dt.datetime.now()

lgbmr = LGBMRegressor(objective='regression', 

#                       num_leaves=4,

                      learning_rate=0.01, 

                      n_estimators=1000,

#                       max_bin=200, 

#                       bagging_fraction=0.75,

#                       bagging_freq=5, 

#                       bagging_seed=7,

#                       feature_fraction=0.2,

#                       feature_fraction_seed=7,

                      verbose=-1

                     )

lgbmr_score = cv_rmse(lgbmr).mean()

endtime = dt.datetime.now()

lgbmr_build_time = endtime-starttime



# Extreme Gradient Boosting

starttime = dt.datetime.now()

xgbr = XGBRegressor(learning_rate=0.01,

                    n_estimators=1000,

#                     max_depth=3,

#                     min_child_weight=0,

#                     gamma=0,

#                     subsample=0.7,

#                     colsample_bytree=0.7,

                    objective='reg:squarederror',

#                     nthread=-1,

#                     scale_pos_weight=1,

#                     reg_alpha=0.00006, 

                    random_state=100

                   )

xgbr_score = cv_rmse(xgbr).mean()

endtime = dt.datetime.now()

xgbr_build_time = endtime-starttime



# Stacking with a XGBoost optimizer (stacking seems pretty interesting)

starttime = dt.datetime.now()

stacking_generate = StackingCVRegressor(regressors=(rf, gbr, lasso_cv, ridge_cv, glmnet_cv, lgbmr, xgbr),

                                        meta_regressor=ridge_cv,

                                        use_features_in_secondary=True)

endtime = dt.datetime.now()

stacking_build_time = endtime-starttime

# stacking_score = cv_rmse(stacking_generate).mean() - value error feature mismatch
# Create a dataframe to store the values and models

initial_scores_df = pd.DataFrame({'CV Score': [lm_score,

                                               rf_score,

                                               gbr_score,

                                               lasso_cv_score,

                                               ridge_cv_score,

                                               glmnet_cv_score,

                                               svr_score,

                                               lgbmr_score,

                                               xgbr_score]})



initial_scores_df.index = ['LM', 'RF', 'GBR', 'LASSO' ,'Ridge' ,'ElasticNet' ,'Support Vector' ,'Light GBM' , 'XGB']

sorted_initial_scores_df = initial_scores_df.sort_values(by='CV Score', ascending=True)

sorted_initial_scores_df
# Fit the new models so use for blending

starttime = dt.datetime.now()



print('Fitting stacking model...')

stack_gen_model = stacking_generate.fit(np.array(X_train), np.array(y_train))

print('Fitting linear model...')

lm_fit_model = lm.fit(X_train, y_train)

print('Fitting forest model...')

rf_fit_model = rf.fit(X_train, y_train)

print('Fitting gradient boosting model...')

gbr_fit_model = gbr.fit(X_train, y_train)

print('Fitting lasso model...')

lasso_fit_model = lasso_cv.fit(X_train, y_train)

print('Fitting ridge model...')

ridge_fit_model = ridge_cv.fit(X_train, y_train)

print('Fitting elastic net model...')

glmnet_fit_model = glmnet_cv.fit(X_train, y_train)

print('Fitting support vector model...')

svr_fit_model = svr.fit(X_train, y_train)

print('Fitting light gradient boosting model...')

lgbr_fit_model = lgbmr.fit(X_train, y_train)

print('Fitting extreme gradient boosting model...')

xgbr_fit_model = xgbr.fit(X_train, y_train)



endtime = dt.datetime.now()



print('Model Fitting Time Elapsed: {}'.format(endtime-starttime))
# Define blending function

def blend_models(X):

    '''

    Function will return predicted values from the test data with

    weighted percentages per model

    '''

    return ( (0.14 * stack_gen_model.predict(np.array(X))) + \

             (0.13 * ridge_fit_model.predict(X)) + \

             (0.13 * lasso_fit_model.predict(X)) + \

             (0.13 * glmnet_fit_model.predict(X)) + \

             (0.11 * gbr_fit_model.predict(X)) + \

             (0.10 * xgbr_fit_model.predict(X)) + \

             (0.09 * lgbr_fit_model.predict(X)) + \

             (0.09 * svr_fit_model.predict(X)) + \

             (0.08 * rf_fit_model.predict(X))

            )
print('RMSE Score:')

print(base_rmse(y_test, blend_models(X_test)))

blended_score = base_rmse(y_test, blend_models(X_test))
scorelist = {}



scorelist['LM'] = lm_score

scorelist['RF'] = rf_score

scorelist['GBR'] = gbr_score

scorelist['Lasso'] = lasso_cv_score

scorelist['Ridge'] = ridge_cv_score

scorelist['Elastic Net'] = glmnet_cv_score

scorelist['Support Vector'] = svr_score

scorelist['Light GBM'] = lgbmr_score

scorelist['XGB'] = xgbr_score

scorelist['Blended'] = blended_score



build_time_list = {'Model': ['Ridge', 'LM', 'Elastic Net', 'Lasso', 'GBR', 'XGB', 'Blended',

                             'Light GBM', 'Support Vector', 'RF', 'Stacking'],

                   

                   'Build Time': [ridge_cv_build_time, lm_build_time, glmnet_cv_build_time,

                                  lasso_cv_build_time, gbr_build_time, lgbmr_build_time,

                                  xgbr_build_time, stacking_build_time]

                  }



model_names = [i for i in scorelist.keys()]

model_scores = [i for i in scorelist.values()]



scorelist_df = pd.DataFrame({'Model': model_names,

                             'Score': model_scores})



scorelist_df.sort_values(by='Score', ascending=True)
# Create figure space

sns.set_style("white")



# Create catplot

ax = sns.catplot(x='Model',

                 y='Score',

                 data=scorelist_df,

                 kind='point',

                 height=6) 



# Set plot features

plt.xticks(rotation='45')

plt.title('Final Model Scores', fontsize=15)

plt.xlabel('Model', fontsize=12)

plt.ylabel('Score (RMSE)', fontsize=12)

plt.show()
# Create figure space

sns.set_style("white")

plt.figure(figsize=(8,3))



# Create catplot

sns.pointplot(x='Model',

                 y='Score',

                 data=scorelist_df,

                 height=6) 



# Set plot features

plt.xticks(rotation='45')

plt.title('Final Model Scores', fontsize=18)

plt.xlabel('Model', fontsize=15)

plt.ylabel('Score (RMSE)', fontsize=15)

plt.show()

import matplotlib.pyplot as pyplot

pyplot.bar(range(len(xgbr_fit_model.feature_importances_)), xgbr_fit_model.feature_importances_)

pyplot.show()
plot_importance(xgbr_fit_model)

pyplot.show()
headers = X_train.columns

len(headers)
# Create a new function to capture feature importance for free models (RF, GB, XGB)

def feature_importance(model):

    

    importance = pd.DataFrame({'Feature': headers,

                               'Importance': np.round(model.feature_importances_,5)})

    

    importance = importance.sort_values(by='Importance', ascending=False).set_index('Feature')

    

    return importance
feature_importance(xgbr_fit_model)
feature_importance(rf_fit_model)
# Store top 50 features from xgb

xgb_top_features = list(feature_importance(xgbr_fit_model).iloc[0:50,].index)



# Store top 50 features from rf

rf_top_features = list(feature_importance(rf_fit_model).iloc[0:50,].index)
# Create unique feature list

# (set(xgb_top_features) | set(rf_top_features))

unique_top_features = sorted(np.unique(xgb_top_features+rf_top_features))

len(unique_top_features)
unique_top_features
# Import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV # RandomizedSearchCV() GridSearchCV



def tune_hyperparameters(model, x, y):

    global best_params, best_score

    '''

    1. Create a grid search on all of the hyperparameters and score them

    2. After creating the grid with the hyperparameters, we need to fit the model with training data

    3. After fitting, get the best parameters and score

    '''

    # Grid search 10-fold cross-validation through hyperparameters

    grid = RandomizedSearchCV(model, verbose=0, cv=10, scoring='neg_mean_squared_error', n_jobs=-1) # optional n_jobs=-1 to use all cores

    

    # Fit the model using the grid

    grid.fit(x, y)

    

    # Get best parameters and scores

    best_params, best_score = grid.best_params_, np.round(grid.best_score_*100, 5)
# Import StandardScaler() from the preprocessing module

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()



# Transform the variables to be on the same scale

X_train_new = scaler.fit_transform(X_train) # 1. fit_transform(training data)

X_test_new = scaler.transform(X_test)       # 2. transform(testing data)



# Build param list  - np.arange(1, 15, 0.5)

r_alphas=[ 1 ,  1.5,  2 ,  2.5,  3 ,  3.5,  4 ,  4.5,  5 ,  5.5,  6 ,

        6.5,  7,  7.5,  8 ,  8.5,  9 ,  9.5, 10 , 10.5, 11 , 11.5,

       12 , 12.5, 13 , 13.5, 14 , 14.5]



ridge_cv_new = RidgeCV(alphas=r_alphas, cv=10)

ridge_cv_new.fit(X_train_new, y_train)

cv_rmse(ridge_cv_new).mean()
l_alphas = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.001, 0.01]



lasso_cv_new = LassoCV(alphas=l_alphas, cv=10, n_jobs=-1)

lasso_cv_new.fit(X_train_new, y_train)

cv_rmse(lasso_cv_new).mean()
lasso_cv_new
# import dill

# dill.dump_session('09-house-regression-env.db')

# dill.load_session('09-house-regression-env.db')