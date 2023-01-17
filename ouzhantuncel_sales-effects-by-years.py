# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

data.head()
data[["LotFrontage","SalePrice"]]
data.plot(kind="scatter", x="LotFrontage",y="SalePrice",marker=5,color="red")

plt.xlabel("LotFrontage")

plt.ylabel("SalePrice")

plt.title("Choice of Usage House")     # The best house choice is sales price between 100000-200000 and lot frontage range 50-100. 
data[["LotFrontage","LotArea"]]
data.plot(kind="scatter",x="LotFrontage",y="LotArea",marker=5,color="g")

plt.xlabel("LotFrontage")

plt.ylabel("LotArea")

plt.title("Effect of Lot Area on Sales House")
data["SalePrice"].mean()
analysisofyear=data[["YrSold","SalePrice"]]

analysisofyear.head()
data["YrSold"].value_counts()
year2006=analysisofyear[analysisofyear["YrSold"]==2006]["SalePrice"].mean()

year2006

year2007=analysisofyear[analysisofyear["YrSold"]==2007]["SalePrice"].mean()

year2007
year2008=analysisofyear[analysisofyear["YrSold"]==2008]["SalePrice"].mean()

year2008
year2009=analysisofyear[analysisofyear["YrSold"]==2009]["SalePrice"].mean()

year2009
year2010=analysisofyear[analysisofyear["YrSold"]==2010]["SalePrice"].mean()

year2010
data[data["YrSold"]==2006]["LotFrontage"].value_counts()
data[data["YrSold"]==2007]["LotFrontage"].value_counts()
data[data["YrSold"]==2008]["LotFrontage"].value_counts()
data[data["YrSold"]==2009]["LotFrontage"].value_counts()
data[data["YrSold"]==2010]["LotFrontage"].value_counts()
data[data["YrSold"]==2006]["LotArea"].value_counts()
data[data["YrSold"]==2007]["LotArea"].value_counts()
data[data["YrSold"]==2008]["LotArea"].value_counts()
data[data["YrSold"]==2009]["LotArea"].value_counts()
data[data["YrSold"]==2010]["LotArea"].value_counts()
dfy=[[2006,year2006,314,174,50,14145,3072],[2007,year2007,329,107,41,40094,6000],[2008,year2008,304,80,32,13125,2887],[2009,year2009,338,149,50,12800,3378],[2010,year2010,175,152,21,13214,4608]]

df=pd.DataFrame(dfy,columns=['Year','Average Sale Price','Number of Sale House ','Max Lot Frontage','Min Lot Frontage','Max Lot Area','Min Lot Area'])

df
data[data["YrSold"]==2006]["MSSubClass"].value_counts()
data[data["YrSold"]==2007]["MSSubClass"].value_counts()
data[data["YrSold"]==2008]["MSSubClass"].value_counts()
data[data["YrSold"]==2009]["MSSubClass"].value_counts()
data[data["YrSold"]==2010]["MSSubClass"].value_counts()