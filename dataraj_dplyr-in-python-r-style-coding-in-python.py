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
!pip install dplython
!pip install plydata
import dplython as dpl

from dplython import (DplyFrame, 

                      X, 

                      diamonds, 

                      dfilter,

                      select,

                      sift, 

                      sample_n,

                      sample_frac, 

                      head, 

                      arrange,

                      mutate,

                      nrow,

                      group_by,

                      summarize, 

                      DelayFunction) 
head(diamonds,3)
type(diamonds)
nrow(diamonds)
priceCutWise = diamonds.groupby("cut")["price"].mean()

priceCutWise
diamondCol = select(diamonds,X.carat, X.cut,X.color,X.clarity,

                                X.depth,X.table,X.price)

diamondCol.head(5)
diamondCol = diamonds >> select(X.carat, X.cut,X.color,X.clarity,

                                X.depth,X.table,X.price)

diamondCol.head(5)
diamonds.describe()
filteredData = dfilter(diamonds, X.carat > 0.5, X.price > 950)

filteredData.head()
# Filtering the data using pipe operators 

filteredData = diamonds >> dfilter( X.carat > 0.5, X.price > 950)

filteredData.head()
sortedData = arrange(diamonds, X.carat, X.cut)

sortedData.head()
groupd = group_by(diamonds,X.cut,X.color)

summaryVal = summarize(groupd,meanval=X.price.mean())

summaryVal
filteredData = sift(diamonds, X.carat > 0.5, X.price > 950)

filteredData.head()
import plydata as pldt
diamondCol = pldt.select(diamonds,"carat", "cut","color","clarity",

                                "depth","table","price")

diamondCol.head(5)
diamondCol = diamonds >> pldt.select("carat", "cut","color","clarity",

                                "depth","table","price")

diamondCol.head(5)
filteredData = pldt.query(diamonds, "carat > 0.5 & price > 950")

filteredData.head()

# Filtering the data using pipe operators 

filteredData = diamonds >> pldt.query("carat > 0.5 & price > 950")

filteredData.head()


sortedData = pldt.arrange(diamonds, "carat", "cut")

sortedData.head()
sortedData = diamonds >> pldt.arrange( "carat", "cut")

sortedData.head()
groupd = pldt.group_by(diamonds,"cut","color")

summaryVal = pldt.summarize(groupd,meanval="np.mean(price)")

summaryVal
summaryVal = diamonds >> pldt.group_by("cut","color") >> pldt.summarize(meanval="np.mean(price)")

summaryVal