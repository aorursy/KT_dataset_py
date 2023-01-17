# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
diamonds=sns.load_dataset("diamonds")

df=diamonds.copy()

df.head()
df.info()
df.describe().T
df.cut.value_counts()
from pandas.api.types import CategoricalDtype
cut_categories=["Fair", "Good", "Very Good", "Premium", "Ideal"]

color_categories=["D","E","F","G","H","I","J"]

clarity_categories=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df.cut=df.cut.astype(CategoricalDtype(categories=cut_categories, ordered=True))

df.color=df.color.astype(CategoricalDtype(categories=color_categories, ordered=True))

df.clarity=df.clarity.astype(CategoricalDtype(categories=clarity_categories, ordered=True))
df.cut.head(1)

df.color.head(1)

df.clarity.head()
df.cut.value_counts().plot.barh().set_title("Cut Değişkeninin Sınıf Frekansları");
df.color.value_counts().plot.barh().set_title("Color Değişkeninin Sınıf Frekansları");
df.clarity.value_counts().plot.barh().set_title("Clarity Değişkeninin Sınıf Frekansları");
sns.barplot(x="cut",y=df.cut.index,data=df);
sns.catplot(x="cut",y="price",data=df);
sns.barplot(x="cut",y="price",hue="color",data=df);
x=df.groupby(["cut","color"])["price"].mean();

x
sns.distplot
sns.distplot(df.price,kde=False)
sns.distplot(df.price);
sns.distplot(df.price,hist=False);

sns.kdeplot(df.price,shade=True);
(sns

 .FacetGrid(df,

            hue="cut",

            height=6,

            xlim=(0,10000))

 .map(sns.kdeplot,

      "price",shade=True)

 .add_legend());
(sns

 .FacetGrid(df,

            hue="color",

            height=6,

            xlim=(0,10000))

 .map(sns.kdeplot,

      "price",shade=True)

 .add_legend());
(sns

 .FacetGrid(df,

            hue="clarity",

            height=6,

            xlim=(0,10000))

 .map(sns.kdeplot,

      "price",shade=True)

 .add_legend());
sns.catplot(x="cut",y="price",hue="color",kind="point",data=df);

sns.catplot(x="cut",y="price",hue="clarity",kind="point",data=df);