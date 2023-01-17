# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# checking miami data
pumpkin_miami = pd.read_csv("../input/miami_9-24-2016_9-30-2017.csv")
print(pumpkin_miami)
import glob
path =r'../input' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)
pumpkin_data = pd.concat(list_)
# full dataframe info
print(pumpkin_data.info())
print(pumpkin_data.head())
print(pumpkin_data.columns)
pumpkin_filtered_data = pumpkin_data.drop(['Appearance','Comments','Condition','Crop','Environment','Grade','Offerings','Price Comment','Quality','Storage','Trans Mode','Commodity Name'], axis =1)
# filtered dataframe info
pumpkin_filtered_data.info()
# firstly find the maximum high price
pumpkin_filtered_data['High Price'].max()
# retrieve all the data for maximum high price
pumpkin_filtered_data[pumpkin_filtered_data['High Price'] == 480.0]
# find the lowest low price
pumpkin_filtered_data['Low Price'].min()
# retrieve all the data for minimum low price
pumpkin_filtered_data[pumpkin_filtered_data['Low Price'] == 0.23999999999999999]
pumpkin_filtered_data.groupby(['Item Size'])['High Price'].mean().plot(kind='bar')
plt.ylabel("Mean High Pumpkin Price")
pumpkin_filtered_data['Item Size'].value_counts()
pumpkin_filtered_data.groupby(['Item Size'])['High Price'].median().plot(kind='bar')
plt.ylabel("Median High Pumpkin Price")
# checking color of pumpkin
pumpkin_filtered_data.groupby(['Color'])['High Price'].mean().plot(kind='bar')
plt.ylabel("Mean High Pumpkin Price")
pumpkin_filtered_data['Color'].value_counts()
pumpkin_filtered_data.groupby(['Color'])['High Price'].median().plot(kind='bar')
plt.ylabel("Median High Pumpkin Price")
pumpkin_filtered_data.groupby(['Package'])['High Price'].mean().plot(kind='bar')
plt.ylabel("Mean High Pumpkin Price")
pumpkin_filtered_data[pumpkin_filtered_data.Package == 'each'].groupby('Item Size')['High Price'].max()
pumpkin_filtered_data.dropna(axis=0, how='all').groupby(['City Name'])['Variety'].nunique().plot(kind='bar')
plt.ylabel("Number of Varieties")
pumpkin_filtered_data.dropna(axis=0, how='all').groupby(['Variety'])['City Name'].nunique().plot(kind='bar')
plt.ylabel("Number of Cities")
pumpkin_filtered_data.groupby(['City Name'])['High Price'].mean().plot(kind = 'bar')
plt.ylabel("Average High Price")
pumpkin_filtered_data.groupby(['City Name'])['Low Price'].mean().plot(kind = 'bar')
plt.ylabel("Average Low Price")
pumpkin_filtered_data['Origin'].nunique()
pumpkin_filtered_data.groupby(['Origin'])['High Price'].mean().plot(kind = 'bar')
plt.ylabel("Average High Price")