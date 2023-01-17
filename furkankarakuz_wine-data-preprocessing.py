# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")

df=df.iloc[:,1:]

df.head()
df.shape
df.info()
df.dtypes
df.describe().T
def missing_value_table(df):

    missing_value = df.isna().sum().sort_values(ascending=False)

    missing_value_percent = 100 * df.isna().sum()//len(df)

    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)

    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})

    cm = sns.light_palette("lightgreen", as_cmap=True)

    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)

    return missing_value_table_return

  

missing_value_table(df)
df.region_2.unique().size
df.region_1.unique().size
df.designation.unique().size
plt.figure(figsize=(15,6))

df.dtypes.value_counts().plot.barh();
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))

sns.kdeplot(df["points"],shade=True,ax=ax[0]);

sns.kdeplot(df["price"],shade=True,ax=ax[1]);
sns.pairplot(df)

sns.set(style="ticks", color_codes=True)
corr = df.corr()

plt.figure(figsize=(15,7))

sns.heatmap(corr, annot=True);
country = df.country.value_counts()[:10]

plt.figure(figsize=(15,7))

sns.barplot(x=country.index, y=country.values, palette="dark")

plt.xticks(rotation='vertical')

plt.ylabel('Frequency')

plt.xlabel('Country')

plt.title('top 10 countries',color = 'darkblue',fontsize=15);
f, ax = plt.subplots(figsize=(15,7))

sns.despine(f, left=True, bottom=True)



sns.scatterplot(x="points", y="price",

                hue="price",

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=df, ax=ax);
num_data=pd.DataFrame(df.dtypes[df.dtypes!="object"]).index
count = 0

fig, ax =plt.subplots(nrows=1,ncols=2, figsize=(20,8))

sns.boxplot(df[num_data[0]],ax=ax[0])

sns.boxplot(df[num_data[1]],ax=ax[1]);
lower_and_upper = {}



for col in num_data:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = 1.5*(q3-q1)

    

    lower_bound = q1-iqr

    upper_bound = q3+iqr

    

    lower_and_upper[col] = (lower_bound, upper_bound)

    df.loc[(df.loc[:,col]<lower_bound),col]=lower_bound*0.75

    df.loc[(df.loc[:,col]>upper_bound),col]=upper_bound*1.25

    

    

lower_and_upper
count = 0

fig, ax =plt.subplots(nrows=1,ncols=2, figsize=(20,8))

sns.boxplot(df[num_data[0]],ax=ax[0])

sns.boxplot(df[num_data[1]],ax=ax[1]);
import missingno as msno
df.isnull().sum()
msno.bar(df);
msno.matrix(df);
msno.heatmap(df);
df["country"].fillna(df["country"].mode()[0],inplace=True)

df["province"].fillna(df[df["country"]=="US"]["province"].mode()[0],inplace=True)
list_columns=["designation","region_1","region_2","variety","winery"]

for i in df[list_columns]:

    df[i].fillna(df[i].mode()[0],inplace=True)
from sklearn.impute import KNNImputer
knn_imputer=KNNImputer()

df["price"]=knn_imputer.fit_transform(df[["price"]])
msno.bar(df);
df.isnull().sum()