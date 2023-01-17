# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

data.head()
data.tail()
data.info()
data.dtypes
def change(variable):

    data[variable]=data[variable].astype("category")

category=["cut","color","clarity"]

for c in category:

    change(c)

print(data.dtypes)
data.shape
data.columns
data.describe().T
data.isnull().values.any()
data.isnull().sum()
kat_df=data.select_dtypes("category")

kat_df.columns

kat_df.head()
def fonc(variable):

    a=kat_df[variable].value_counts()

    b=kat_df[variable].value_counts().count()

    print(a)

    print("Number of classes of categorical variables {}".format(b))

variable=["cut","color","clarity"]

for c in variable:

    fonc(c)

    

    
kat_df["cut"].value_counts().plot.barh()
data_num=data.select_dtypes(include=["int64","float64"])

data_num.head()
data_num.describe().T
(data["cut"]

 .value_counts()

 .plot.barh()

 .set_title("Cut"));
sns.barplot(x=data.cut,y=data.cut.index,data=data);
sns.catplot(x="cut",y="price",data=data);
sns.barplot(x="cut",y="price",hue="color",data=data);
data.groupby(["cut","color"])["price"].mean()
sns.distplot(data["price"],kde=False,bins=10,color="red");
sns.distplot(data.price,hist=False,color="purple");
sns.kdeplot(data.price,shade=True);
(sns

 .FacetGrid(data,

              hue="cut",

              height=5,

              xlim=(0,10000))

 .map(sns.kdeplot,"price",shade=True)

 .add_legend()

);
sns.catplot(x="cut",y="price",hue="color",kind="point",data=data);
sns.boxplot(data["price"]);