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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os

import pandas as pd

import seaborn as sns
print(os.listdir('../input/google-landmarks-dataset'))

baseLoc = '../input/google-landmarks-dataset/'
nRowsRead = 10000 # specify 'None' if want to read whole file

# index.csv has 1098461 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv(baseLoc + 'index.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'index.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head()
nRowsRead = 10000 # specify 'None' if want to read whole file

# test.csv has 117703 rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv(baseLoc + 'test.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'test.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head()
nRowsRead = 10000 # specify 'None' if want to read whole file

# train.csv has 1225029 rows in reality, but we are only loading/previewing the first 1000 rows

df3 = pd.read_csv(baseLoc + 'train.csv', delimiter=',', nrows = nRowsRead)

df3.dataframeName = 'train.csv'

nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
df3.head()
train_data = df3.copy()

test_data = df2.copy()
print(len(train_data['url'].unique()))

print(len(train_data['id'].unique()))
#Downloading the images 

from IPython.display import Image

from IPython.core.display import HTML 

def display_image(url):

    img_style = "width: 500px; margin: 0px; float: left; border: 1px solid black;"

    #images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])

    image=f"<img style='{img_style}' src='{url}' />"

    display(HTML(image))

display_image(train_data['url'][155])
# now open the URL

temp = 155

print('id', train_data['id'][temp])

print('url:', train_data['url'][temp])

print('landmark id:', train_data['landmark_id'][temp])
train_data['landmark_id'].value_counts()
train_data.isnull().sum()
test_data.isnull().sum()
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

len(train_data['landmark_id'].unique())
print("Number of classes under 20 occurences",(train_data['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train_data['landmark_id'].unique()))
train_data['landmark_id'].value_counts().keys()
from IPython.display import Image

from IPython.core.display import HTML 



def display_category(urls):

    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"

    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])



    display(HTML(images_list))
category = train_data['landmark_id'].value_counts().keys()[0]

urls = train_data[train_data['landmark_id'] == category]['url']

display_category(urls)
urls = train_data['url']

l1 = [url.split('/')[2] if len(url.split('/'))>2 else '' for url in urls ]

l1
train_data['site'] = pd.Series(l1)
train_data['site']
urls = test_data['url']

l2 = [url.split('/')[2] if len(url.split('/'))>2 else '' for url in urls ]

test_data['site'] = pd.Series(l2)

test_data.head()
train_site = pd.DataFrame(train_data.site.value_counts())

test_site = pd.DataFrame(test_data.site.value_counts())
train_site
test_site
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
