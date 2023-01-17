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
import matplotlib.pyplot as plt

import seaborn as sns



amazon = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv", encoding="ISO-8859-1")
amazon.describe()
amazon.info()
amazon.head()

# looks like "month" is not in English 
amazon["month"].unique()
amazon["month"] = np.where(amazon["month"] == "Janeiro", "Jan", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Fevereiro", "Feb", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Março", "Mar", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Abril", "Apr", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Maio", "May", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Junho", "June", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Julho", "July", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Agosto", "Aug", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Setembro", "Sep", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Outubro", "Oct", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Novembro", "Nov", amazon["month"])

amazon["month"] = np.where(amazon["month"] == "Dezembro", "Dec", amazon["month"])



# change all month names into English forms
amazon.head()
############################## Exploratory Data Analysis ##################################



df_state = amazon.groupby("state").agg({"number":"sum"}).sort_values("number", ascending=False).reset_index()



plt.figure(figsize = (16,8))

sns.barplot(x="state", y="number", data=df_state)

plt.xticks(rotation=90)

plt.show()

# it clearly shows that Mato Grosso state has highest number of fires reported

# perhaps it is because Mato Grosso is mostly covered with Amazon rainforest.
df_year = amazon.groupby("year").agg({"number":"sum"}).reset_index()



plt.figure(figsize = (16,8))

sns.barplot(x="year", y="number", data=df_year)

plt.show()



# it seems that year 2003 has the highest number of forest fires reported
df_month = amazon.groupby("month").agg({"number":"sum"}).reset_index()



plt.figure(figsize = (16,8))

sns.barplot(x="month", y="number", data=df_month)

plt.show()



# it seems that July has the highest number of forest fires reported
df_mato_year = amazon[amazon["state"] == "Mato Grosso"].groupby("year").agg({"number":"sum"}).reset_index()



plt.figure(figsize = (16,8))

sns.barplot(x="year", y="number", data=df_mato_year)

plt.title("Forest Fire at Mato Grosso")

plt.show()



# it seems that year 2009 has the highest forest fires reported at Mato Grosso
df_mato_month = amazon[amazon["state"] == "Mato Grosso"].groupby("month").agg({"number":"sum"}).sort_values("number", ascending=False).reset_index()



plt.figure(figsize = (16,8))

sns.barplot(x="month", y="number", data=df_mato_month)

plt.title("Forest Fire at Mato Grosso")

plt.show()



# it seems September has the least number forest fires reported