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
df=pd.read_csv("../input/Summary of Weather.csv")
df.head(3)
df2=pd.read_csv("../input/Weather Station Locations.csv")
df2.head(5)
df.describe()
df=df.drop(['WindGustSpd','PoorWeather','PRCP','PGT','TSHDSBRSGF' ,'FT','FB','FTI','ITH','SD3','RHX','RHN','RVG','WTE','DR','SPD','SND'],axis=1)
df.isnull().sum()
df.describe()

MAX_fill=df.MAX.median()
df.MAX.fillna(MAX_fill,inplace=True)
df.isnull().sum()
MIN_fill=df.MIN.median()
df.MIN.fillna(MIN_fill,inplace=True)
MEA_fill=df.MEA.median()

SNF_fill=df.SNF.median()