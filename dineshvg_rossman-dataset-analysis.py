# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np

## Print the first five rows of the training data to view the existing columns from which features can be calculated.
train_df = pd.read_csv('../input/train.csv')
# Converting the date object to a date-time object for easier sorting.
train_df['Date'] =pd.to_datetime(train_df.Date)
train_df.sort_values(by=['Date']).head()
# We can see that there are days on which there was no sale. This would not help with the analysis.
# Removing the training data where sales did not take place.
train_df = train_df[train_df.Sales > 0]
len(train_df)
# Calculating the Average sale per customer in the modified trainging set.
train_df['AvgSalePerCustomer'] = train_df['Sales']/train_df['Customers']
train_df.head()
# statistics
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
cdf = ECDF(train_df['AvgSalePerCustomer'])
plt.plot(cdf.x, cdf.y, label = "statmodels");
plt.xlabel('Sale per Customer');
plt.show()
# The store gives us more information about the store and granular information about the items on sale.
store_df = pd.read_csv('../input/store.csv')
store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace = True)
# replace NA's by 0
store_df.fillna(0, inplace = True)

## Convert Assortment, store type and PromoInterval as numbers to use as features
# a = 0, b = 1, c = 3, d = 4.
store_df['StoreType'] = store_df['StoreType'].apply({'a':0, 'b':1, 'c':2, 'd':3 }.get)
store_df['Assortment'] = store_df['Assortment'].apply({'a':0, 'b':1, 'c':2, 'd':3 }.get)
store_df['PromoInterval'] = store_df['PromoInterval'].apply({0:0, 'Jan,Apr,Jul,Oct':1, 'Mar,Jun,Sept,Dec':2}.get)
store_df.head()
## Plot competition distance based on store type.

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
plt.figure()
x = store_df['StoreType']
y1 = store_df['CompetitionDistance']
plt.title('Store analysis')
plt.xlabel('Store type')
plt.ylabel('CompetitionDistance')
plt.scatter(x,y1)
plt.show()
train_merge = pd.merge(train_df, store_df, on='Store')
train_merge.head()
## Plot sale/customer for different dates.

import matplotlib.pyplot as plt

mpl.rcParams['agg.path.chunksize'] = 10000
plt.figure()
x = train_merge['Date']
y1 = train_merge['AvgSalePerCustomer']/max(train_merge['AvgSalePerCustomer'])
plt.title('Average sales per customer over time')
plt.xlabel('Date')
plt.ylabel('Scaled Avgerage Sale per Customer')
plt.plot(x,y1)
plt.show()
## Sales seem to be high during the end of the year. This could be because of promotions or holidays.

train_merge['Year'] = pd.DatetimeIndex(train_merge['Date']).year 
train_merge['Month'] = pd.DatetimeIndex(train_merge['Date']).month 
train_merge = train_merge.drop(['Date'], axis=1)
plt.figure()
data_2013 = train_merge[train_merge.Year==2013]
x = data_2013['Month']
y1 = data_2013['AvgSalePerCustomer']/max(data_2013['AvgSalePerCustomer'])
plt.title('Average sales per customer in 2013')
plt.xlabel('Month')
plt.ylabel('Scaled Avgerage Sale per Customer')
plt.plot(x,y1)
plt.show()
train_merge = train_merge[train_merge.Open==1]
train_merge['CompetitionDistance'].isnull().values.any()
train_merge['CompetitionDistance'].fillna(0, inplace=True)

plt.figure()
plt.plot(train_merge['CompetitionDistance'], train_merge['AvgSalePerCustomer']/max(train_merge['AvgSalePerCustomer']))
plt.title('Sales based on competition distance')
plt.xlabel('Competition distance')
plt.ylabel('AvgSalePerCustomer')
plt.show()
train_merge['StoreType'].isnull().values.any()
train_merge['StoreType'].fillna(0, inplace=True)

plt.figure()
plt.plot(train_merge['StoreType'], train_merge['AvgSalePerCustomer']/max(train_merge['AvgSalePerCustomer']))
plt.title('Sales based on types of stores')
plt.xlabel('Store types')
plt.ylabel('AvgSalePerCustomer')
plt.show()

## We see that store type c and d have more sales than store type a and b.
train_merge['StateHoliday'] = train_merge['StateHoliday'].apply({'0':0, 'a':1, 'b':2, 'c':3, 0:0 }.get)
print(train_merge['StateHoliday'].unique())
plt.figure()
plt.plot(train_merge['StateHoliday'], train_merge['AvgSalePerCustomer']/max(train_merge['AvgSalePerCustomer']))
plt.title('Sales based on types of StateHoliday')
plt.xlabel('StateHoliday')
plt.ylabel('AvgSalePerCustomer')
plt.show()
plt.figure()
x = train_merge['DayOfWeek']
y1 = train_merge['AvgSalePerCustomer']/max(train_merge['AvgSalePerCustomer'])
plt.title('Average sales per customer for different DayOfWeek')
plt.xlabel('Month')
plt.ylabel('Scaled Avgerage Sale per Customer')
plt.scatter(x,y1)
plt.show()
# feature set is finally ready after pre-processing.
X_train = train_merge
X_train = X_train.drop(['Sales'], axis=1)
X_train =X_train.drop(['Customers'], axis=1)
X_train.fillna(0, inplace = True)
X_train.head()
# Principal component analysis for feature reduction

from sklearn.decomposition import PCA

nf = 2
pca = PCA(n_components=nf)
# X is the matrix transposed (n samples on the rows, m features on the columns)
X_new = pca.fit_transform(X_train)
X_new.shape
y_train = train_merge['Sales']
y_train.fillna(0, inplace = True)
y_train.shape
# Classification

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

clf_rf = RandomForestClassifier()
clf_nb = GaussianNB()
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20, criterion='entropy', splitter='random'),
                        algorithm="SAMME.R")
from sklearn.model_selection import train_test_split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_new, y_train, test_size=0.75, random_state=42)