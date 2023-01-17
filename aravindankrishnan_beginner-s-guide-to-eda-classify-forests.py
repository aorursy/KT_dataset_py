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
# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Import training data - review shape and columns

import pandas as pd

train = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

print(train.info())
for column in train.columns[10:]:

    print(train[column].value_counts())
# get the index of start and end columns for wilderness area and soil type. we will convert these individual OHE columns into categorical variables.

print(train.columns.get_loc('Wilderness_Area1'))

print(train.columns.get_loc('Wilderness_Area4'))

print(train.columns.get_loc('Soil_Type1'))

print(train.columns.get_loc('Soil_Type40'))
# Reverse the one hot encoding for Wilderness Area and Soil Type to create one column each for Wilderness Areas and Soil Types which contain all all possible category values

train_1 = train.copy()

train_1['Wild_Area'] = train_1.iloc[:,10:14].idxmax(axis=1)

train_1['Soil_Type'] = train_1.iloc[:,14:54].idxmax(axis=1)

train_1.head()

# Drop the one hot encoded columns to a get a data frame only with numeric and the categorical columns

col_to_drop = np.arange(10,54)

col_to_drop

train_1.drop(train_1.columns[col_to_drop], axis = 1, inplace = True)

train_1.dtypes
# Look at analysis friendly dataframe which has numeric and category variables to do some EDA

train_1.head()
train.iloc[:,:10].describe()
print(train_1.Cover_Type.value_counts())
print(train_1.Wild_Area.value_counts())
sns.set(style="whitegrid")

plt.figure(figsize=(5,3))

sns.barplot(train_1.Wild_Area.value_counts().index,y=train_1.Wild_Area.value_counts())

plt.xticks(rotation=90)
print(train_1.Soil_Type.value_counts())
sns.set(style="whitegrid") # To show grid lines to match numbers.

plt.figure(figsize=(15,5))

sns.barplot(x=train_1.Soil_Type.value_counts().index,y=train_1.Soil_Type.value_counts())

plt.xticks(rotation=90)
plt.figure(figsize=(20,10))

sns.countplot('Soil_Type', data = train_1, hue = 'Cover_Type', dodge=False)

plt.xticks(rotation=90)

plt.legend(loc='upper right', ncol = 3, frameon=False)
len(train_1.Soil_Type.value_counts())
# list of non zero soil types available in training data

avail_soil = train_1.Soil_Type.value_counts().index

print(avail_soil)



# Get all possible Soil Types available as features in the original training data as columns

all_soil = train.columns[14:54]

print(all_soil)

# Check the missing soil types from training data by comparing all_soil and avail_soil

miss_soil = np.setdiff1d(all_soil,avail_soil)

print(miss_soil)
# Plot histograms and kde customizing various options for hist, kde and rug. plot in a loop using range of i

for i in range(0,len(train_1.columns[:10])):

    plt.figure(figsize=(15,10))

    plt.subplot(5,2,i+1)

    sns.distplot(train_1.iloc[:,i],rug=True,kde_kws={'color':'r','lw':3, 'label':'KDE'}, hist_kws={'color':'g', 'linewidth':5, 'histtype':'step','label':'HIST'})

    plt.show()
# Plot histograms and kde customizing various options for hist, kde and rug. plot in a loop using range of i

plt.figure(figsize=(15,10))

for i in range(0,len(train_1.columns[:10])):

    plt.subplot(5,2,i+1)

    sns.distplot(train_1.iloc[:,i],rug=True,kde_kws={'color':'r','lw':3, 'label':'KDE'}, hist_kws={'color':'g', 'linewidth':5, 'histtype':'step','label':'HIST'})

plt.tight_layout()

# Plot histograms colored by cover type and overlapping with no hist bars

for column in train_1.columns[:10]:

    plt.figure(figsize=(35,15))

    g=sns.FacetGrid(train_1,hue='Cover_Type')

    g.map(sns.distplot,column, hist=False, label='Cover_Type')

    plt.legend(frameon=False)

    plt.show()
# Break down histograms colored by cover type and overlapping with no hist bars. Facet by Wild Area to see the impact of different wilderness area.

for column in train_1.columns[:10]:

    plt.figure(figsize=(35,15))

    g=sns.FacetGrid(train_1,hue='Cover_Type', col='Wild_Area')

    g.map(sns.distplot,column, hist=False, label='Cover_Type')

    for ax in g.axes.ravel():

        ax.legend()

    plt.show()
# Create an ordered list for Soil Type values to sort the facet plot.

soil_type_order = train.columns[14:54]

soil_type_order
# Break down histograms colored by cover type and overlapping with no hist bars. Facet by Soil Type to see the impact of different Soil Types.

for column in train_1.columns[:1]:

    plt.figure(figsize=(10,10))

    g=sns.FacetGrid(train_1,hue='Cover_Type', col='Soil_Type', col_order=soil_type_order, col_wrap=3, height=4, aspect = 1.5)

    g.map(sns.distplot,column, hist=False, label='Cover_Type')

    for ax in g.axes.ravel(): 

        ax.legend(frameon=False)

        ax.set_xlabel(column)

        plt.ylim(0,0.02)

    plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(data=train_1, palette='Set2')

plt.xticks(rotation=90)
plt.figure(figsize=(15,10))

for i,column in enumerate(train_1.columns[:10]):

    plt.subplot(2,5,i+1)

    #g=sns.FacetGrid(train_1)

    #g.map(sns.boxplot,y=train_1[column])

    sns.boxplot(y=column,data = train_1)

plt.tight_layout()
plt.figure(figsize=(20,10))

for i,column in enumerate(train_1.columns[:10]):

    plt.subplot(5,2,i+1)

    sns.boxplot(train_1[column], palette = 'Set2')

        

plt.tight_layout() # use this to show up any hidden labels when the plots are originally drawn overlapping.
plt.figure(figsize=(20,20))

for i,column in enumerate(train_1.columns[:10]):

    plt.subplot(5,2,i+1)

    sns.boxplot(x='Cover_Type', y = column, data = train_1 )

plt.tight_layout()
# Create a function to calculate number and % of outliers based on 1.5 times IQR as the cut off on both sides of the IQR (viz. 75th and 25th quantiles)

def detect_outliers_perc(df,column, ratio = 1.5):

    lower,upper = np.percentile(df[column],25),np.percentile(df[column],75)

    iqr = upper-lower

    lower_bar,upper_bar = (lower - (ratio*iqr)),(upper + (ratio*iqr))

    outliers = (df[column] < lower_bar) | (df[column] > upper_bar)

    print(column,':','No of Outliers:',np.sum(outliers),';','% of outliers:',(np.sum(outliers)*100/train_1.shape[0]))



for column in train_1.columns[:10]:

    detect_outliers_perc(train_1,column)
# Create a copy df from train_1 so as to use the train_1 df for all EDA without impact of outliers.

train_2 = train_1.copy()

# Discard rows containing even one outlier for any of the features on that row.

def remove_outliers(df,column,lowperc=25,highperc=75,ratio=1.5):

    lower,upper = np.percentile(df[column],lowperc),np.percentile(df[column],highperc)

    iqr = upper-lower

    lower_bar,upper_bar = (lower - (ratio*iqr)),(upper + (ratio*iqr))

    outliers = (df[column] < lower_bar) | (df[column] > upper_bar)

    #print('Number of Outliers:',np.sum(outliers))

    df = df[~outliers]

    print('After Removing outliers for',column,'New Shape is:',df.shape, 'Outlier rows removed:',np.sum(outliers))

    

    return df

print('Original shape of Data frame is:',train_1.shape)   

for column in train_2.columns[:10]:

    train_2 = remove_outliers(train_2,column, ratio = 1.5)

print('Final Shape of Data frame is: ',train_2.shape)
plt.figure(figsize=(20,10))

for i,column in enumerate(train_2.columns[:10]):

    plt.subplot(5,2,i+1)

    sns.boxplot(train_2[column], palette = 'Set2')

        

plt.tight_layout() # use this to show up any hidden labels when the plots are originally drawn overlapping.
sns.pairplot(train_1, hue = 'Cover_Type', palette = sns.color_palette('Set1',7))
sns.scatterplot(x='Slope', y = 'Elevation', hue = 'Cover_Type', data = train_1)
plt.figure(figsize=(15,15))

for i,column in enumerate(train_1.columns[1:10]):

    #plt.figure(figsize=(6,6))

    plt.subplot(3,3,i+1)

    sns.scatterplot(x='Elevation', y = column, hue = 'Cover_Type', data = train_1, palette = sns.color_palette('Set1',7), alpha = 0.2)

    plt.legend(frameon=False,loc='best', bbox_to_anchor = (1,0.5))

    plt.tight_layout()

    
plt.figure(figsize=(10,6))

sns.heatmap(train_1.corr(),annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # check for seaborn divergent color palettes to use for cmap.
plt.figure(figsize=(10,6))

import numpy as np

corr = train_1.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

#with sns.axes_style("white"):

sns.heatmap(corr,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0, mask = mask)
