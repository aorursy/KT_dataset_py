# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

data.info()
data.head()
HDIvaluetofill = data["HDI for year"][np.logical_not(data["HDI for year"].isnull())].mean()

data["HDI for year"] = data["HDI for year"].fillna(HDIvaluetofill)
data.iloc[:,9] = data.iloc[:,9].str.replace(",","")
data.iloc[:,9] = data.iloc[:,9].astype(int)
data.head()
def bar_plot(data,features):

    for i in features:

        x = data[i].value_counts().index

        y = data[i].value_counts().values

        dataout = pd.DataFrame({"Yil":x,"Miktar":y})

        indexes = (dataout.Miktar.sort_values(ascending= False)).index.values

        newdata = dataout.reindex(indexes).reset_index(drop=True)

        

        plt.figure(figsize=(25,5))

        sns.barplot(x=newdata.Yil,y=newdata.Miktar, order=newdata.Yil)

        plt.xticks(rotation = 60)

        plt.show()

listtoshow=["year"]

bar_plot(data,listtoshow)
data1 = data.groupby(["country"],as_index=False)[["year"]].mean().sort_values(by="year",ascending=False)



plt.figure(figsize=(25,5))

sns.barplot(data1.country,data1.year)

plt.xticks(rotation=90)

plt.show()