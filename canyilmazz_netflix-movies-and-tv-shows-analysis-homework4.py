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
# data frames from dictionary

team=["Juventus","Barcelona"]

player=["CRonaldo","LMessi"]

list_label=["team","player"]

list_col=[team,player]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
#Add new columns

df["Age"]=[35,33]

df
import matplotlib.pyplot as plt

data=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

data.head()
data.shape
data.info()
#Plotting all data

data1=data.loc[:,["show_id","release_year"]]

data1.plot()
#subplots

data1.plot(subplots=True)
data.plot(subplots=True)
#scatter plot

data1.plot(kind="scatter",x="show_id",y="release_year")

plt.show()
data.describe().T
time_list=["1998-02-26","1996-11-13"]

print(type(time_list[1]))

datetime_list=pd.to_datetime(time_list)

print(type(datetime_list))
data2=data.head()

date_list=["2013-12-17","2013-12-18","2013-12-19","2013-12-20","2013-12-21"]

datetime_list=pd.to_datetime(date_list)

data2["date"]=datetime_list

data2=data2.set_index("date")

data2