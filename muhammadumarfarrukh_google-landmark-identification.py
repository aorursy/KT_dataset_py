from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os

import pandas as pd

import seaborn as sns
print(os.listdir('../input'))
# Histogram of column data

def plotHistogram(df, nHistogramShown, nHistogramPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nHistRow = (nCol + nHistogramPerRow - 1) / nHistogramPerRow

    plt.figure(num=None, figsize=(6*nHistogramPerRow, 8*nHistRow), dpi=80, facecolor='w', edgecolor='k')

    for i in range(min(nCol, nHistogramShown)):

        plt.subplot(nHistRow, nHistogramPerRow, i+1)

        df.iloc[:,i].hist()

        plt.ylabel('counts')

        plt.xticks(rotation=90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

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

nRowsRead = 10000 # specify 'None' if want to read whole file

# index.csv has 1098461 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/index.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'index.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotHistogram(df1, 10, 5)
nRowsRead = 10000 # specify 'None' if want to read whole file

# test.csv has 117703 rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('../input/test.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'test.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotHistogram(df2, 10, 5)
nRowsRead = 10000 # specify 'None' if want to read whole file

# train.csv has 1225029 rows in reality, but we are only loading/previewing the first 1000 rows

df3 = pd.read_csv('../input/train.csv', delimiter=',', nrows = nRowsRead)

df3.dataframeName = 'train.csv'

nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
plotHistogram(df3, 10, 5)
train_data = df3

test_data = df2



print("Training data size:",train_data.shape)

print("Test data size:",test_data.shape)
train_data.head()

train_data['url'][33]
#Displaying number of unique URLs & ids

len(train_data['url'].unique())

len(train_data['id'].unique())
#Downloading the images 

from IPython.display import Image

from IPython.core.display import HTML 

def display_image(url):

    img_style = "width: 500px; margin: 0px; float: left; border: 1px solid black;"

    #images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])

    image=f"<img style='{img_style}' src='{url}' />"

    display(HTML(image))
#Displaying the images

display_image(train_data['url'][155])
# now open the URL

temp = 155

print('id', train_data['id'][temp])

print('url:', train_data['url'][temp])

print('landmark id:', train_data['landmark_id'][temp])
train_data['landmark_id'].value_counts().hist()
# missing data in training data 

total = train_data.isnull().sum().sort_values(ascending = False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)

missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
# missing data in test data 

total = test_data.isnull().sum().sort_values(ascending = False)

percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)

missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test_data.head()
# Occurance of landmark_id in decreasing order(Top categories)

temp = pd.DataFrame(train_data.landmark_id.value_counts().head(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
# Plot the most frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
# Occurance of landmark_id in increasing order

temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(8))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id','count']

temp
 #Plot the least frequent landmark_ids

plt.figure(figsize = (9, 8))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

plt.show()
# Unique URL's

train_data.nunique()
#Class distribution

plt.figure(figsize = (10, 8))

plt.title('Category Distribuition')

sns.distplot(train_data['landmark_id'])



plt.show()
print("Number of classes under 20 occurences",(train_data['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train_data['landmark_id'].unique()))
from IPython.display import Image

from IPython.core.display import HTML 



def display_category(urls, category_name):

    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"

    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])



    display(HTML(images_list))
category = train_data['landmark_id'].value_counts().keys()[0]

urls = train_data[train_data['landmark_id'] == category]['url']

display_category(urls, "")
category = train_data['landmark_id'].value_counts().keys()[1]

urls = train_data[train_data['landmark_id'] == category]['url']

display_category(urls, "")
# Extract repositories names for train data

ll = list()

for path in train_data['url']:

    ll.append(train_data['url'].str.split('/').str[2])

train_data['site'] = ll

train_data['site'] 
ll = list()

for path in test_data['url']:

    ll.append(test_data['url'].str.split('/').str[2])

test_data['site'] = ll
print("Train data shape -  rows:",train_data.shape[0]," columns:", train_data.shape[1])

print("Test data size -  rows:",test_data.shape[0]," columns:", test_data.shape[1])
train_data.head()
train_site = pd.DataFrame(train_data.site.value_counts())

test_site = pd.DataFrame(test_data.site.value_counts())
train_site
# Plot the site occurences in the train dataset

trsite = pd.DataFrame(list(train_site.index),train_site['site'])

trsite.reset_index(level=0, inplace=True)

trsite.columns = ['Count','Site']

plt.figure(figsize = (6,6))

plt.title('Sites storing images - train dataset')

sns.set_color_codes("pastel")

sns.barplot(x = 'Site', y="Count", data=trsite, color="blue")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()
# Plot the site occurences in the test dataset

tesite = pd.DataFrame(list(test_site.index),test_site['site'])

tesite.reset_index(level=0, inplace=True)

tesite.columns = ['Count','Site']

plt.figure(figsize = (6,6))

plt.title('Sites storing images - test dataset')

sns.set_color_codes("pastel")

sns.barplot(x = 'Site', y="Count", data=tesite, color="magenta")

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

plt.show()