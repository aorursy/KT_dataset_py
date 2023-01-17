# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
df.head(5) #veri setinin ilk 5 item ini getir.
df.dtypes #verilerimizin tiplerini inceledik.ve 1 adet  int,8 adet float türünde değişkenlerimiz var.
df.shape #verimiz 3997697 satır dan oluşuyor zaten 8 adet de veri tipi mevcut
df.columns
df.describe().T #veri setini düzgün şekilde listeleyebilmek için transpozunu aldım.Bu tabloda column larımın max min değerlerini ortalamalarını ve standart sapma değerlerini inceledim.
df.isnull().values.any() #hiç eksik gözlem var mı?
df.isnull().sum() #hangi değişkende kaçar tane var?

df["Open"].fillna(0,inplace=True) #open değişkenindeki eksik gözlemleri 0 ile doldur.
df["High"].fillna(0,inplace=True) #high değişkenindeki eksik gözlemleri 0 ile doldur.
df["Low"].fillna(0,inplace=True) #low değişkenindeki eksik gözlemleri 0 ile doldur.
df["Close"].fillna(0,inplace=True) #close değişkenindeki eksik gözlemleri 0 ile doldur.
df["Volume_(BTC)"].fillna(0,inplace=True) #Volume_(BTC) değişkenindeki eksik gözlemleri 0 ile doldur.
df["Volume_(Currency)"].fillna(0,inplace=True) #Volume_(Currency değişkenindeki eksik gözlemleri 0 ile doldur.
df["Weighted_Price"].fillna(0,inplace=True) #Weighted_Price değişkenindeki eksik gözlemleri 0 ile doldur.
df.isnull().sum()

import seaborn as sns

sns.distplot(df.Open,kde=False); #open değişkeninin histogram grafiği
sns.distplot(df.Close,kde=False); #close değişkeninin histogram grafiği
#histogram ve yoğunluk grafiği
sns.distplot(df.Open);
sns.distplot(df.High,kde=False); #High değişkeninin histogram grafiği
sns.distplot(df.Low,kde=False); #Low değişkenin Histogram grafiği
sns.distplot(df.Weighted_Price,kde=False);
sns.kdeplot(df.Open,shade=True);
sns.boxplot(x=df["Open"]); #Open değişkeninin kutu grafiği oluşturulması
sns.boxplot(x=df["Close"]);
sns.boxplot(x=df["High"]);
sns.boxplot(x=df["Low"]);
sns.boxplot(x=df["Volume_(BTC)"]);
sns.boxplot(x=df["Volume_(Currency)"]);
sns.heatmap(df); #ısı grafiği
sns.lineplot(x="Open",y="Low",data=df);