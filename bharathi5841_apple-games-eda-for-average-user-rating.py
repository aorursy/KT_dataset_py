# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Loading the Data
#Dataset is at location : /kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv

Load_Full=pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')
L_apple_games=Load_Full.copy()

L_apple_games.head()
L_apple_games['In-app Purchases'].value_counts()
L_apple_games['Primary Genre'].value_counts()
L_apple_games['Subtitle'].value_counts()
L_apple_games=L_apple_games.drop(['URL','Icon URL','Primary Genre','Subtitle'],axis=1)
L_apple_games.head(2)
L_apple_games['Name'].value_counts()
L_apple_games['ID'].isnull().value_counts()
L_apple_games['Average User Rating'].value_counts()
L_apple_games['Average User Rating'].value_counts().sort_values(ascending=True)
L_apple_games['User Rating Count'].value_counts()
L_apple_games.shape
L_apple_games.info()
L_apple_games=L_apple_games[~L_apple_games['Average User Rating'].isna()]
L_apple_games.info()
tmp=L_apple_games[['User Rating Count','Price','Size']]
#
tmp[tmp.columns[0]].head(1)
plt.figure(figsize=(16,16))
plt.subplot(3,2,1)
sns.distplot(tmp[tmp.columns[0]],hist=True,kde=True)
plt.subplot(3,2,3)
sns.distplot(tmp[tmp.columns[1]],hist=True,kde=False)
plt.subplot(3,2,5)
sns.distplot(tmp[tmp.columns[2]],hist=True,kde=False)
for j in range(3):
    plt.subplot(3,2,2*j+2)
    sns.boxplot(y=tmp[tmp.columns[j]])
    plt.title(tmp.columns[j] + "Boxplot")


L_apple_games.head(3)
#L_apple_games['Price'].value_counts()
#L_apple_games['Languages'].value_counts()
#np.log(L_apple_games.size)
#L_apple_games['Genres'].value_counts()
#L_apple_games['In-app Purchases'].fillna(0).map({0:0}).fillna(1)
#L_apple_games['In-app Purchases']=pd.to_numeric(L_apple_games['In-app Purchases'],errors='coerce')
pd.Timestamp("today").dt.days

L_apple_games['Is_Paid']= L_apple_games.Price.map({0:0}).fillna(1)
L_apple_games['Num_Languages']=L_apple_games.Languages.str.count(",")+1
L_apple_games['Num_Genres']=L_apple_games.Genres.str.count(",")+1
L_apple_games['has_In_app_purchase']= L_apple_games['In-app Purchases'].fillna(0).map({0:0}).fillna(1)
L_apple_games['age_of_app']=(pd.Timestamp("today")-pd.to_datetime(L_apple_games['Original Release Date'])).dt.days
L_apple_games['time_since_update']=(pd.Timestamp("today")-pd.to_datetime(L_apple_games['Current Version Release Date'])).dt.days
L_apple_games.head(10)
plt.figure(figsize=(36,28))
plt.subplot(431)
sns.countplot(L_apple_games['Average User Rating'])
plt.title("Average User Rating")
plt.subplot(432)
sns.countplot(L_apple_games['Is_Paid'])
plt.title("Is_Paid")
plt.subplot(433)
sns.countplot(L_apple_games['has_In_app_purchase'])
plt.title("has_In_app_purchase")
plt.subplot(434)
sns.distplot(L_apple_games["Num_Languages"].dropna())
plt.title("Num_Languages Histogram")
plt.subplot(435)
sns.countplot(L_apple_games["Num_Genres"].dropna())
plt.title("Num_Genres Histogram")
plt.subplot(4,3,6)
sns.distplot(L_apple_games["age_of_app"].dropna())
plt.title("age_of_app Histogram")

plt.subplot(4,3,7)
sns.distplot(L_apple_games["time_since_update"].dropna())
plt.title("time_since_update Histogram");
plt.figure(figsize=(24,7))

plt.subplot(1,2,1)
tb1=pd.crosstab(L_apple_games['Is_Paid'],L_apple_games['Average User Rating'])
tb2=(tb1.T/tb1.T.sum(axis=0)).T
sns.heatmap(tb1,cmap='plasma',square=True)

plt.subplot(1,2,2)
for_bar=tb1.reset_index().melt(id_vars=['Is_Paid'])
sns.barplot(x=for_bar['Average User Rating'],y=for_bar['value'],hue=for_bar['Is_Paid'])