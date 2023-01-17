# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/StudentsPerformance.csv')
data1.columns
data1.info()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,vmin = 0,vmax= 1)

plt.show()
data1.columns = [each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in data1.columns]
data1.columns
# scatter plot  

data1.plot(kind = "scatter",x="math_score",y = "reading_score",color = "red")

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="math_score",y = "writing_score",color = "blue")

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="reading_score",y = "writing_score",color = "green")

plt.show()
data1.describe()
data2 = data1.loc[:,["math_score","reading_score","writing_score"]]

data2.plot()
data0 = data1.head(7)

date_list = ["2017-01-10","2016-02-10","2017-03-10","2017-03-15","2018-03-16","2018-02-17","2018-03-17"]

datetime_object = pd.to_datetime(date_list)

data0["date"] = datetime_object

data0= data0.set_index("date")

data0 
print(data0.loc["2018-03-17"])

print(data0.loc["2017-01-10":"2017-03-15"])
data0.resample("A").mean()
data0.resample("M").max()
data0.resample("M").first().interpolate("linear")
data0.resample("M").max().interpolate("linear")
data1['test_preparation'][200]
data1.test_preparation[500]
data1.loc[1,['test_preparation']]
data1[["gender","test_preparation"]]
data1.loc[490:500,"race/ethnicity":"parental_level"]
data1.columns
data1.parental_level = [i.split()[0]+"_"+i.split()[1] if len(i.split()) > 1 else i for i in data1.parental_level]
data1.parental_level
filter1 = data1.math_score > 90

filter2 = data1.reading_score > 85

filter3 = data1.writing_score > 85

data1[filter1 & filter2 & filter3]
data1.gender[data1.math_score > 99]
data1.gender[data1.reading_score > 99]
data1.gender[data1.writing_score > 99]
data1.gender[filter1 & filter2 & filter3]
def app(n):

    return n-1

data = data1.math_score.apply(app)

data
data1.reading_score.apply(lambda n : n-1)
data1['average_score'] = (data1.math_score + data1.reading_score + data1.writing_score)/3.0

data1
print(data1.index.name)
data1.index.name = "student"

data1.head()
data1.index = range(1,1001,1)

data1.head()
data1.groupby('gender').mean()
data1.groupby("race/ethnicity").math_score.max()
data1.groupby("race/ethnicity").reading_score.max()
data1.groupby("race/ethnicity").writing_score.max()
data1.groupby("race/ethnicity").mean()
data1.groupby("race/ethnicity")[["math_score","average_score"]].min()