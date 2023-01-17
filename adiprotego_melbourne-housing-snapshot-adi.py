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
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')

df.head()
plt.title('Distribution of Price Data')

sns.distplot(df['Price'],kde=True)

print("Skewness: %f" % df['Price'].skew())
PLog = np.log(df['Price'])

PLog.skew()
target=PLog

plt.title('Distribution of the skewed distribution')

sns.distplot(target,kde=True)
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
p_corr=df.corr()

p_corr['Price'].sort_values(ascending=False)
miss=df.isnull().sum()

miss.sort_values(ascending=False)
sns.set(font_scale=1)

plt.figure(figsize=(10,10))

miss.plot.barh(title='Missing Values')

percent_missing=df.isnull().mean()*100

percent_missing.sort_values(ascending=False)
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.title('Distribution of Year Built data')

sns.kdeplot(df['YearBuilt'],shade=True,color='red')

plt.subplot(2,2,2)

sns.boxplot('YearBuilt',data=df,color='green')
df['YearBuilt'].describe()
df['YearBuilt'].replace({np.nan:df['YearBuilt'].median()},inplace=True)
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.title('Distribution of Building Area data')

sns.kdeplot(df['BuildingArea'],shade=True,color='red')

plt.subplot(2,2,2)

sns.boxplot('BuildingArea',data=df,color='blue')
df['BuildingArea'].describe()
df['BuildingArea'].replace({np.nan:df['BuildingArea'].mode()},inplace=True)
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.title('Distribution of Car data')

sns.distplot(a=df['Car'],kde=False)

plt.subplot(2,2,2)

sns.boxplot('Car',data=df,color='blue')
df['Car'].describe()
df['Car'].fillna(0,inplace=True)
plt.figure(figsize=(10,10))

ca_count=sns.countplot(df['CouncilArea'])

ca_count.set_xticklabels(ca_count.get_xticklabels(),rotation=90);
df['CouncilArea'].replace({np.nan:'Unavailable'},inplace=True)
df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

features_to_convert=['Car','Bedroom2','Bathroom','Rooms']

for i in features_to_convert:

    df[i]=df[i].astype(object)
categoric=df.select_dtypes(include='object')

numeric=df.select_dtypes(exclude=['object'])

numeric.info()
sns.scatterplot(df['Distance'],target)
sns.scatterplot(df['Postcode'],target)
sns.scatterplot(df['Landsize'],target)

plt.xlim(-1,100)
sns.scatterplot(x=df['Landsize'],y=target,hue=df['Type'])

plt.xlim(-1,100000)
sns.scatterplot(df['BuildingArea'],target,data=df)

plt.xlim(0,1000)
sns.scatterplot(df['Longtitude'],target)

categoric.info()
house_features=df[['Rooms','Bedroom2','Bathroom','Car']]

plt.figure(figsize=(20,10))

n=1

for i in house_features:

    plt.subplot(2,2,n)

    x=df.groupby([house_features[i]])['Price'].median().sort_values()

    ax=sns.boxplot(x=house_features[i],y='Price',data=df,order=list(x.index),palette='Blues')

    ax=sns.stripplot(x=house_features[i],y='Price',data=df,color='red',size=1.5)

    plt.xlabel(i)

    n+=1

        





    
plt.figure(figsize=(10,10))

df['Date']=pd.to_datetime(df['Date'])

df['year'] = pd.DatetimeIndex(df['Date']).year

sns.boxenplot(x='year',y=target,data=df)
sns.violinplot(x='Type',y=target,data=df)

r_plot=sns.boxplot(x='Regionname',y=target,data=df)

r_plot.set_xticklabels(r_plot.get_xticklabels(),rotation=90);
sns.boxplot(x='Method',y=target,data=df)
plt.figure(figsize=(10,10))

sns.boxplot(x=target,y='CouncilArea',data=df)
sns.set()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.title('Costliest suburbs')

df.groupby(["Suburb"])['Price'].median().sort_values(ascending=False)[:10].plot.bar()

plt.subplot(2,2,2)

plt.title('Cheapest suburbs')

df.groupby(["Suburb"])['Price'].median().sort_values(ascending=True)[:10].plot.bar()

sns.set()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)

plt.title('Expensive sellers')

df.groupby(["SellerG"])['Price'].median().sort_values(ascending=False)[:10].plot.bar()

plt.subplot(2,2,2)

plt.title('Cheaper sellers')

df.groupby(["SellerG"])['Price'].median().sort_values(ascending=True)[:10].plot.bar()
