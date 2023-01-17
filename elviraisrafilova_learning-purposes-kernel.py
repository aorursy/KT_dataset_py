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
import pandas as pd

import numpy as np

import shapefile as shp

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool
#BUILDING DATA FRAMES FROM SCRATCH

# data frames from dictionary

bookttitle = ["Korku","Bilinmeyen Bir Kadının Mektubu", "Gece Uçuşu", "Savaş Pilotu"]

author = ["Stefan Zweig","Stefan Zweig","Saint Exupery", "Saint Exupery"]

list_label = ["bookttitle","author"]

list_col = [bookttitle,author]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#add new column

df["numberofpages"] = ["70","66","108", "175"]

df
# Broadcasting

df["bookrating"] = 0 #Broadcasting entire column

df
df.dtypes
df["bookrating"] = df["bookrating"].astype('float')

df["numberofpages"] = df["numberofpages"].astype('int')
df.dtypes

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")



date_list = ["1992-01-10","1992-02-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

df["date"] = datetime_object

# lets make date as index

df= df.set_index("date")

df 
print(df.loc["1993-03-16"])

print(df.loc["1992-01-10":"1993-03-15"])
df.resample("A").mean()
df.resample("M").mean()
#with interpolate

# We can interpolete from first value

df.resample("M").first().interpolate("linear")
df.resample("M").mean().interpolate("linear")
#PIVOTING DATA FRAMES

dic = {"chewinggum":["wrigeys","wrigeys","doublemint","doublemint","juicyfruit", "juicyfruit"],"market":["carrefour","migros","carrefour","migros", "carrefour","migros"],"price":[3,3.2,3.5,4,2.9,3.7],"dailysales":[2000,4000,7200,6500,3700,1000]}

df = pd.DataFrame(dic)

df
df.pivot(index="chewinggum",columns = "market",values="price")
df1 = df.set_index(["chewinggum","market"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="chewinggum",value_vars=["price","dailysales", "market"])
df.groupby("chewinggum").mean()
df.groupby("chewinggum").price.max()
df.groupby("chewinggum")[["price","dailysales", "market"]].min() 
df.info()