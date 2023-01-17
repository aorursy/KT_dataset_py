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
data=pd.read_csv('/kaggle/input/car-sale-advertisements/car_ad.csv',encoding ='iso-8859-9')
import matplotlib.pyplot as plt

import seaborn as sns
data.shape
data.isnull().sum()
data.dropna(axis=0,inplace=True)
data.isnull().sum()
data.dtypes
data.shape
data=data[:150]
sns.relplot(x="price", y="engV", data=data)
sns.relplot(x="price", y="engV", data=data).set_xticklabels(rotation=30)

sns.relplot(x="mileage", y="price", data=data)
sns.relplot(x="mileage", y="price", data=data)
sns.relplot(x="year", y="price", data=data).set_xticklabels(rotation=30)
sns.relplot(x="year", y="mileage", hue="engType", data=data).set_xticklabels(rotation=30);
data.head(5)
sns.relplot(x="year", y="mileage", hue="car", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage", style="drive",hue="drive", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",style="body", hue="body", data=data).set_xticklabels(rotation=30);
sns.relplot(x="mileage", y="price",style="drive",hue="engType",data=data).set_xticklabels(rotation=30)
sns.relplot(x="year", y="price", size="drive",data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="price", size="drive",sizes=(15, 200),data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="price", kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="price",sort=False ,kind="line", data=data).set_xticklabels(rotation=30);

data.head(5)
sns.relplot(x="year", y="mileage", kind="line", data=data).set_xticklabels(rotation=30);

sns.relplot(x="year", y="mileage", kind="line", data=data).set_xticklabels(rotation=30);

sns.relplot(x="year", y="mileage",ci=None,kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",ci="sd",kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",estimator=None,kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",hue="drive",kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",hue="drive",kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",hue="drive",style="body",kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",hue="drive",style="body",markers=True,kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="mileage",hue="drive",style="body",dashes=False,markers=True,kind="line", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year", y="price", hue="drive",

            col="body", data=data).set_xticklabels(rotation=30);
sns.relplot(x="year",y="mileage",hue="body",col="drive",row="engType",height=3,kind="line", estimator=None,data=data).set_xticklabels(rotation=30);