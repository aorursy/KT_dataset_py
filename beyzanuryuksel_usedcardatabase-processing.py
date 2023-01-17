# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv( "../input/used-cars-database/autos.csv",encoding = "ISO-8859-1")

data.head()
data.drop("dateCrawled",axis=1,inplace=True) 
data.drop("dateCreated",axis=1,inplace=True) 
data.drop("lastSeen",axis=1,inplace=True)
data.drop("nrOfPictures",axis=1,inplace=True)
data.drop("postalCode",axis=1,inplace=True)


df = pd.DataFrame(data)
print(df.shape)
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.hist(figsize=(20,16))
sns.pairplot(df)
sns.catplot(x = "monthOfRegistration" , y = "price" , data=df);

sns.barplot(x = "yearOfRegistration" , y = df.yearOfRegistration.index , data=df);
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['price'], df['kilometer'])
ax.set_xlabel('price')
ax.set_ylabel('kilometer')
plt.show()
corrmat = df.corr() 
  
f, ax = plt.subplots(figsize =(12, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
def boxplt(df, columns):
    for i in df.columns:
        if(df[i].dtype== np.int64 or df[i].dtype== np.float64):
            fig, ax =plt.subplots(figsize=(8,6))
            sns.boxplot(x = df[i])
boxplt(data,df.columns)    

def find_outliers(df, columns):
    for i in df.columns:
        if(df[i].dtype== np.int64):
            q1=df[i].quantile(0.25)
            q3=df[i].quantile(0.75)
            iqr=q3-q1
            upper_bound= q3 + 1.5 * iqr
            lower_bound= q1 - 1.5 * iqr
            for k in range(len(df[i])):
                if df[i].iloc[k]<lower_bound:
                    df[i].iloc[k]=lower_bound*0.75
                if df[i].iloc[k]>upper_bound:
                    df[i].iloc[k]=upper_bound*1.25
find_outliers(data,df.columns)
#controling function of find_outlier 
def boxplt(df, columns):
    for i in df.columns:
        if(df[i].dtype== np.int64 or df[i].dtype== np.float64):
            fig, ax =plt.subplots(figsize=(8,6))
            sns.boxplot(x = df[i])
boxplt(data,df.columns)   
#percent of missing values
def percent_missing(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    gf=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    gf= gf[gf["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    fig=sns.barplot(gf.index, gf["Percent"],color="purple",alpha=0.6)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Percent of missing values', fontsize=16)
    plt.title('Missing values of feature', fontsize=16)
    return gf

percent_missing(df)
sns.heatmap(df.isnull(),yticklabels='auto',cmap= "viridis")
#count for group of features
index=[1,2,4,5,7,12,14]
for i in range(len(index)):
    print(df.groupby(df.columns[index[i]])["price"].count())
    print (f"***************\n")
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df[columns] = imp_mean.fit_transform(df[columns])
df.isnull().sum()
