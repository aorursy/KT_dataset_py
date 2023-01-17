# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# First importing seaborn library
import seaborn as sns
#Creating a data frame
DF = pd.read_csv("../input/911.csv")
# Head of data frame
DF.head()
#Info of data frame
DF.info()
# no. of unique title names
DF["title"].nunique()
# splitting of title
DF["SR"]=DF["title"].apply(lambda title : title.split(":")[0])
DF.head()
DF["SR"].value_counts().head(10)
# plotting a graph using seaborn
sns.countplot(x = "SR", data = DF)
# converting the format of timestamp
DF["timeStamp"] = pd.to_datetime(DF["timeStamp"])
DF.head()
DF["hours"] = DF["timeStamp"].apply(lambda time : time.hour)
DF["month"] = DF["timeStamp"].apply(lambda time : time.month)
DF["day"] = DF["timeStamp"].apply(lambda time : time.dayofweek)
sns.countplot(x="month", data = DF, hue = "SR")
byMonth = DF.groupby("month").count()
byMonth.head()
byday = DF.groupby("day").count()
byday.head()
byday["twp"].plot()
byMonth["twp"].plot()
