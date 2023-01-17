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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

sns.set()

import scipy.stats as ss

import missingno as msno



from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer

from sklearn.preprocessing import LabelEncoder



#no boundation on number of columns 

pd.set_option('display.max_columns', None)



#no boundation on number of rows

pd.set_option('display.max_rows', None)



# run multiple commands in a single jupyter cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#Heatmap

def heatmap(df):

    

    '''

    this function takes data frame a input and returns the

    heatmap as output.

    

    Arguments

    ====================

    df : Dataframe or Series 

    

    

    Returns

    ===========

    heatmap

    '''

    corr = df.corr()   #create a correlation df

    fig,ax = plt.subplots(figsize = (30,30))   # create a blank canvas

    colormap = sns.diverging_palette(220,10, as_cmap=True)   #Generate colormap

    sns.heatmap(corr, cmap=colormap, annot=True, fmt='.2f')   #generate heatmap with annot(display value) and place floats in map

#    plt.xticks(range(len(corr.columns)), corr.columns);   #apply  xticks(labels of features)

#    plt.yticks(range(len(corr.columns)), corr.columns)   #apply yticks (labels of features)

    plt.show()

    

    



def categorical_feature_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):

    '''

    

    helper function that gives a quick summary of a given column of categorical data

    

    Arguments

    =============

    dataframe: pandas df

    x: str, horizontal axis to plot the label of categorical data

    y: str, vertical xis to plot hte label of the categorical data

    hue: str, if you want to comparer it to any other variable

    palette: array-like, color of the graph/plot

    

    

    Returns

    ==============

    Quick summary of the categorical data

    

    

    '''

    

    if x==None:

        column_interested = y

        

    else:

        column_interested = x

    series = dataframe[column_interested]

    print(series.describe())

    print('mode', series.mode())

    

    if verbose:

        print('='*80)

        print(series.value_counts())

        

        

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette = palette)

    plt.show()

    

    '''

    

    categorical_summarized does is it takes in a dataframe, together with some input arguments and 

    outputs the following:

        1. The count, mean, std, min, max, and quartiles for numerical data or

        2. the count, unique, top class, and frequency of the top class for non-numerical data.

        3. Class frequencies of the interested column, if verbose is set to 'True'

        4. Bar graph of the count of each class of the interested column

    

    '''
tr  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

tt = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print('train shape:',tr.shape)

print('test shape:', tt.shape)
print('train size:', tr.size)

print('test size:', tt.size)
print('Head:')

tr.head()  # prints first five instances of the dataset

print('Tail:')

tr.tail()  #prints last five instances of the dataset

print('5 Random samples from the train dataset:')

tr.sample(5)  #5 random samples of the dataset
print('Columns of the train dataset:')

tr.columns
print('columns in train but not in test:')

tr.columns.difference(tt.columns)
print('colummns in common train and test')

(tr.columns).intersection(tt.columns)
print('Information on train dataframe:')



tr.info()
print('description of the training dataset:')

tr.describe()
tr.isnull().sum().sort_values(ascending=False)
print('total null values:', tr.isna().sum().sum(), 

      'out of', tr.size, '(total entries)' )
tr_num = tr._get_numeric_data()
missing_bar = msno.bar(tr_num)
missing_data_matrix = msno.matrix(tr_num, )
train_missing_data_heatmap = msno.heatmap(tr)
missing_dendrogram = msno.dendrogram(tr)
print('Dataframe of Features with numerical features')

tr_num = tr._get_numeric_data()

tr_num.head()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



tr_cat = tr.select_dtypes(exclude = numerics)

print('dataframes with categorical features:')

tr_cat.head()
print('list of features with numerical values:')

tr_num.columns



print('list of features with categorical values:')

tr_cat.columns
print('Shape of the numerical DataFrame:', tr_num.shape)

print('Shape of the Categorical DataFrame:', tr_cat.shape)
# Start with an empty canvas

ax,fig = plt.subplots(1,3,figsize=(20,5))



#first plot - disprinbution plot

dist = sns.distplot(tr_num['SalePrice'],bins=50, ax = fig[0])

title = dist.set_title('Distribution Plot');



#second plot - box plot

box = sns.boxplot(x = tr_num['SalePrice'], linewidth=2.5, ax=fig[1])

title_0 = box.set_title('Boxplot')



#third plot - violin plot

violin = sns.violinplot(x=tr_num['SalePrice'], ax=fig[2])

title_1 = violin.set_title('Violin Plot')



plt.show()
print("Average Price of the House:", tr_num['SalePrice'].mean())

print("Most of the houses are close to" ,tr_num['SalePrice'].mode())

print('The median house price is close to'  , tr_num['SalePrice'].median())
num_features = tr_num.columns





print('Skewness:', tr.SalePrice.skew())

print('Kurtosis:', tr.SalePrice.kurt())
numerical_correlation = tr_num.corr()

numerical_correlation
heatmap(tr_num)
tr_num.corr()['SalePrice'].sort_values(ascending=False)
features_for_corr = {'OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath'}

corr_some_features = tr[features_for_corr].corr()

ax,fig = plt.subplots(figsize=(10,10))

colormap = sns.diverging_palette(220,10, as_cmap=True)

heatmap_some_features = sns.heatmap(corr_some_features, cmap = colormap, annot=True, annot_kws={'size':20}, fmt='.2f')

plt.show()
for i in features_for_corr:

    ax,fig = plt.subplots(1,3, figsize=(20,5))

    box = sns.boxplot(y = tr_num[i], linewidth=2.5, ax=fig[2])

    box_title = box.set_title('Box Plot')

    violin = sns.violinplot(y = tr_num[i], linewidth=2.5, ax=fig[0])

    violin_title  = violin.set_title('Violin Plot')

    dist = sns.distplot(tr_num[i], ax=fig[1], vertical=True)

    distplot_title = dist.set_title('Distribution Plot')
for i in features_for_corr:

    skewness = np.array(tr_num[features_for_corr].skew())

    kurtosis = np.array(tr_num[features_for_corr].kurt())

    mean = np.array(tr_num[features_for_corr].mean())

    median = np.array(tr_num[features_for_corr].median())

    variance = np.array(tr_num[features_for_corr].var())

    

    data_frame = pd.DataFrame({'skewness':skewness,

                               'kurtosis':kurtosis, 

                               'Mean':mean,

                               'Median':median, 

                               'variance':variance},

                              

                              index=features_for_corr,

                              columns={'skewness',

                                       'kurtosis',

                                       'Mean',

                                       'Median',

                                       'variance'})

print(data_frame)
tr_cat = tr.select_dtypes(include = 'object')

tr_cat.head()
for i in tr_cat.columns:

    categorical_feature_summarized(tr_cat, x=i)