# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import re
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df.head(10)
print(df.info())  #veriler hakkında genel bilgi
print(df.columns) #sütun adları
print(df.describe())  #int türündeki verilerin ortalama min ve max değerleri
def bar_plot(variable): #bar plot kullanarak sayısal değerlerin çizimini yapacağız
    var=df[variable]
    varValue=var.value_counts()
    
    plt.figure(figsize=(15,4))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("frekans") #y ekseninin adı
    plt.title(variable)  #başlık ismi
    plt.show()   # plot gösterme kodu
    print("{}: \n {}".format(variable,varValue))
kategori1 = ["condition","cylinders","fuel","transmission","drive","size","type","paint_color","state"] # bar plotla çizim yapacağımız kategoriler
for c in kategori1:  #for döngüsüyle kategori1 de ki verileri döndürüyoruz
        bar_plot(c)
def plot_hist(variable):
    plt.figure(figsize = (15,4))
    plt.hist(df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["lat","long"]
for n in numericVar:
    plot_hist(n)

df = df[df.year > 1985]
df.shape
plt.figure(figsize=(15, 13))
ax = sns.countplot(x = 'year', data=df, palette='Set1')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10);
plt.figure(figsize=(15, 13))
ax = sns.countplot(x = 'manufacturer', data=df, palette='Set1')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=10)
g = sns.factorplot(x = "paint_color", y = "price", data = df, kind = "bar", size = 10)
g.set_ylabels("price")
plt.show()
g = sns.factorplot(x = "fuel", y = "price", data = df, kind = "bar", size = 5)
g.set_ylabels("price")
plt.show()
g = sns.factorplot(x = "condition", y = "price", data = df, kind = "bar", size = 5)
g.set_ylabels("Price")
plt.show()
g = sns.factorplot(x = "cylinders", y = "price", data = df, kind = "bar", size =8)
g.set_ylabels("price")
plt.show()
g = sns.factorplot(x = "transmission", y = "price", data = df, kind = "bar", size =5)
g.set_ylabels("Price")
plt.show()
g = sns.factorplot(x = "drive", y = "price", data = df, kind = "bar", size =7)
g.set_ylabels("Price")
plt.show()
g = sns.factorplot(x = "size", y = "price", data = df, kind = "bar", size =7)
g.set_ylabels("Price")
plt.show()
g = sns.factorplot(x = "type", y = "price", data = df, kind = "bar", size =15)
g.set_ylabels("Price")
plt.show()
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
df.loc[detect_outliers(df,["year","price","odometer","county","lat","long"])]
df.head(10)
# drop outliers
df = df.drop(detect_outliers(df,["year","price","odometer","county","lat","long"]),axis = 0).reset_index(drop = True)
df_len = len(df)
df = pd.concat([df],axis = 0).reset_index(drop = True)
df.head()
df.columns[df.isnull().any()]
df.isnull().sum()