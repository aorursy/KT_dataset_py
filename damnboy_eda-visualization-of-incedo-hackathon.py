# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # for data vizualization

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train= pd.read_csv('../input/train_file.csv')
df_train.head() # first 5 rows
df_train.info() #basic information about the whole data
df_train.describe(include='all') # explicit details about the data 
df_train.Greater_Risk_Question.value_counts()
df_train.isnull().sum()#checking the null values for each variable
df_test= pd.read_csv('../input/test_file.csv')

df_test.head()
df_train.columns # checking the number of columns in the dataset
df_train.nunique()#checking all the unique items in the dataset
df_train.dtypes #checking the datatypes in the data, as we will need to hotencode the data for prediction purpose
# correlation plot of the Variables in the data



f, ax = plt.subplots(figsize = (14, 10))



corr = df_train.corr()

sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 

            cmap = sns.diverging_palette(3, 3, as_cmap = True), square = True, ax = ax)
# visualising the distribution of Drug-abuse in the dataset



df_train['LocationDesc'].value_counts(normalize = True)

df_train['Greater_Risk_Question'].value_counts(dropna = False).plot.bar(color = 'c', figsize = (10, 8))



plt.title('distribution of Drug-intake & the residence')

plt.xlabel('Drug-intake')

plt.ylabel('residence')

plt.show()
# visualising the distribution of Drug-intake & the Age  in the dataset



df_train['LocationDesc'].value_counts(normalize = True)

df_train['Race'].value_counts(dropna = False).plot.bar(color = 'green', figsize = (10, 8))



plt.title('distribution of Drug-intake & the Age')

plt.xlabel('Drug-intake')

plt.ylabel('Race')

plt.show()
#checking the unique values in the Drug_User Column

df_train['Greater_Risk_Question'].value_counts()
#checking the count of People who are addicted in terms of their Grade

df_train['LocationDesc'].value_counts(normalize = True)

df_train['Grade'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 8))

plt.title('people addicted to drugs in terms of their education')

plt.xlabel('Grade')

plt.ylabel('count')

plt.show()
#Visualizing the YEAR Distribution in the Dataset

df_train['YEAR'].value_counts(normalize = True)

df_train['YEAR'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (7, 5))



plt.title('Distribution of 141 coutries in suicides')

plt.xlabel('YEAR')

plt.ylabel('count')

plt.show()
df_train['YEAR'].nunique()
#Visualizing the Location Distribution in the Dataset

%time

df_train['LocationDesc'].value_counts(normalize = True)

df_train['LocationDesc'].value_counts(dropna = False).plot.bar(color = 'purple', figsize = (17, 10))



plt.title('Distribution of 141 coutries in suicides')

plt.xlabel('LocationDesc')

plt.ylabel('count')

plt.show()
#Visualizing the Location Distribution in the Dataset

%time

df_train['GeoLocation'].value_counts(normalize = True)

df_train['GeoLocation'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (17, 10))



plt.title('Distribution of geoloaction ')

plt.xlabel('Geolocation')

plt.ylabel('count')

plt.show()
#Visualizing the Location Distribution in the Dataset

%time

df_train['GeoLocation'].value_counts(normalize = True)

df_train['GeoLocation'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (17, 10))



plt.title('Distribution of geoloaction ')

plt.xlabel('Geolocation')

plt.ylabel('count')

plt.show()
#Visualizing the choice of Drug Abuse in the Dataset

%time

df_train['Greater_Risk_Question'].value_counts(normalize = True)

df_train['Greater_Risk_Question'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))



plt.title('Distribution of Greater_Risk_Question ')

plt.xlabel('Greater_Risk_Question')

plt.ylabel('count')

plt.show()
# Distribution of Addiction based on Gender



%time

df_train['Sex'].value_counts(normalize = True)

df_train['Sex'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))



plt.title('Distribution of gender ')

plt.xlabel('Sex')

plt.ylabel('count')

plt.show()

#total number of addicted people in the dataset

df_train['Description'].value_counts()
# Distribution of Description of Addiction

%time

df_train['Description'].value_counts(normalize = True)

df_train['Description'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))



plt.title('Distribution of Description ')

plt.xlabel('Sex')

plt.ylabel('count')

plt.show()