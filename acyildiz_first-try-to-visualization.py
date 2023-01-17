# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Summary of Weather.csv')
data.sample(5)
data.info()
data.describe()
a = ['Date','MaxTemp','MinTemp','MeanTemp',]
data = data[a]
data
data.tail()
#Separating year by year for examine average , max and min temperature during WWII
first_year = data.loc[0:365]
first_year_max_temp = first_year['MaxTemp'].max()
second_year = data.loc[366:730]
third_year = data.loc[731:1095]
fourth_year = data.loc[118887:119039]
first_year.Date = '1942'
second_year.Date = '1943'
third_year.Date = '1944'
fourth_year.Date = '1945'
frames = [first_year, second_year,third_year,fourth_year ]
newdata = pd.concat(frames)
newdata.index = range(len(newdata))
x = newdata.Date
newdata
#Visualization

plt.figure(figsize=(15,10))
sns.barplot(x=x, y=newdata.MaxTemp)
plt.xticks(rotation= 90)
plt.xlabel('Years')
plt.ylabel('Max Temperature')
plt.title("Max Temperature Of years")
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x=x,y=y,data=newdata,color='lime',alpha=0.8, )
plt.xlabel('Year')
plt.ylabel('Max Temperature')
plt.title("Max Temperature During WWII")
plt.show()

