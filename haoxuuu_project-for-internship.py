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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
import random

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
data.head()
print(data.shape)
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
data.dropna(how ='any', inplace = True)
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
print(data.shape)
data.columns
data.info()
print( len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())
g = sns.countplot(x="Category",data=data, color = 'deepskyblue')
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each category',size = 20)
data['Type'].unique()
labels =data['Type'].value_counts(sort = True).index
sizes = data['Type'].value_counts(sort = True)


colors = ["deepskyblue","salmon"]

 
plt.figure(figsize=(7, 7))
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=270,)

plt.title('Percent of Free App in store',size = 20)
plt.show()
data['Content Rating'].unique()
data[data['Content Rating']=='Unrated']
data = data[data['Content Rating'] != 'Unrated']
g = sns.countplot(x="Content Rating",data=data, color = 'deepskyblue')
print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())
data.Genres.value_counts().head(10)
data.Genres.value_counts().tail(10)
data['Genres'] = data['Genres'].str.split(';').str[0]
print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())
data.Genres.value_counts().tail(10)
data['Genres'].replace('Music & Audio', 'Music',inplace = True)
data = data[data['Genres'] != 'February 11, 2018']
data.Genres.value_counts().tail(10)
g = sns.countplot(x="Genres",data=data, color = 'deepskyblue')
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each Genres',size = 20)
data['Last Updated'].head()
data['new'] = pd.to_datetime(data['Last Updated'])
data['new'].describe()
data['new'].max()
data['new'][0] -  data['new'].max()
data['lastupdate'] = (data['new'] -  data['new'].max()).dt.days
data['lastupdate'].head()
sns.distplot(data['lastupdate'], color = 'deepskyblue')
data['Rating'].describe()
# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.distplot(data.Rating, color = 'deepskyblue')
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)
data['Reviews'].head()
# convert to int

data['Reviews'] = data['Reviews'].apply(lambda x: int(x))
g = sns.kdeplot(data.Reviews, color="deepskyblue")
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)
data['Size'].head()
data['Size'].unique()
len(data[data.Size == 'Varies with device'])
data['Size'].replace('Varies with device', np.nan, inplace = True ) 
data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)
g = sns.kdeplot(data.Size, color="deepskyblue")
g.set_xlabel("Size")
g.set_ylabel("Frequency")
plt.title('Distribution of Size',size = 20)
data['Installs'].head()
data['Installs'].unique()
data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
data.Installs = data.Installs.apply(lambda x: int(x))
data['Installs'].unique()
Sorted_value = sorted(list(data['Installs'].unique()))
data['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
data['Installs'].head()
g = sns.distplot(data.Installs, color = 'deepskyblue')
plt.title('Distribution of Install',size = 20)
data['Price'].head()
data.Price.unique()
data['Price'].value_counts().head(30)
data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))
data['Price'].describe()
g = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10 ,
color = 'lightpink')
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Boxplot of Rating VS Category',size = 20)
g = sns.catplot(x="Type",y="Rating",data=data, kind="box", color = 'lightpink')
g.despine(left=True)
g = g.set_ylabels("Rating")
plt.title('Boxplot of Type VS Rating',size = 20)
g = sns.catplot(x="Content Rating",y="Rating",data=data, kind="box", height = 10 ,color = 'lightpink')
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Box plot Rating VS Content Rating',size = 20)
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').head(1)

data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').tail(1)
g = sns.catplot(x="Genres",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxenplot of Rating VS Genres',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="lastupdate", y="Rating", color = 'deepskyblue',data=data );
plt.title('Rating  VS Last Update( days ago )',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'deepskyblue',data=data[data['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Size", y="Rating", color = 'deepskyblue',data=data);
plt.title('Size VS Reveiws',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'deepskyblue',data=data);
plt.title('Rating VS Installs',size = 20)
plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'deepskyblue',data=data);
plt.title('Scatter plot Rating VS Price',size = 20)
data.loc[ data['Price'] == 0, 'PriceBand'] = '0 Free'
data.loc[(data['Price'] > 0) & (data['Price'] <= 0.99), 'PriceBand'] = '1 cheap'
data.loc[(data['Price'] > 0.99) & (data['Price'] <= 2.99), 'PriceBand']   = '2 not cheap'
data.loc[(data['Price'] > 2.99) & (data['Price'] <= 4.99), 'PriceBand']   = '3 normal'
data.loc[(data['Price'] > 4.99) & (data['Price'] <= 14.99), 'PriceBand']   = '4 expensive'
data.loc[(data['Price'] > 14.99) & (data['Price'] <= 29.99), 'PriceBand']   = '5 too expensive'
data.loc[(data['Price'] > 29.99), 'PriceBand']  = '6 FXXXing expensive'
data[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()
g = sns.catplot(x="PriceBand",y="Rating",data=data, kind="boxen", height = 10 ,color = 'lightpink')
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxen plot Rating VS PriceBand',size = 20)
data.head()
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.dropna(inplace = True)
# for dummy variable encoding for Categories
df = pd.get_dummies(df, columns=['Category'])
#scaling and cleaning size of installation
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None

df["Size"] = df["Size"].map(change_size)

#filling Size which had NA
df.Size.fillna(method = 'ffill', inplace = True)
#Cleaning no of installs classification
df['Installs'] = [int(i[:-1].replace(',','')) for i in df['Installs']]
#Cleaning of content rating classification
RatingL = df['Content Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
df['Content Rating'] = df['Content Rating'].map(RatingDict).astype(int)
#Converting Type classification into binary
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1

df['Type'] = df['Type'].map(type_cat)
#Cleaning prices
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price

df['Price'] = df['Price'].map(price_clean).astype(float)
# convert reviews to numeric
df['Reviews'] = df['Reviews'].astype(int)
#dropping of unrelated and unnecessary items
df.drop(labels = ['Last Updated','Current Ver','Android Ver','App'], axis = 1, inplace = True)
df.info()
#to add into results_index for evaluation of error term 
def Evaluationmatrix_dict(y_true, y_predict, name = 'RFR - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true,y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true,y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true,y_predict)
    return dict_matrix
#for evaluation of error term and 
def Evaluationmatrix(y_true, y_predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(y_true,y_predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(y_true,y_predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(y_true,y_predict)))
X_d = df.drop(labels = ['Rating','Genres'],axis = 1)
y_d = df.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = RandomForestRegressor()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

#evaluation
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test_d,Results_d),orient = 'index')
resultsdf = resultsdf.transpose()

print('预测结果平均值 :'+ str(Results_d.mean()))
print('真实结果平均值 :'+ str(y_test_d.mean()))
print('预测结果方差 :'+ str(Results_d.std()))
print('真实结果方差 :'+ str(y_test_d.std()))
resultsdf.head()
