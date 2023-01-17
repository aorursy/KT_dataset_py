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
dataframe=pd.read_csv("../input/2017.csv")
dataframe.info()
#quick general information about data content
dataframe.dtypes
#Information data types in data
dataframe.columns
#Column in data contains header information
dataframe.head(10)
#According to the survey, the 10 most happiest countries in the world
dataframe.tail(10)
#According to the survey, the world's 10 most unhappy countries
dataframe.loc[149 : 154 , "Country"]
dataframe.describe()
dataframe.loc[0:10,["Country","Happiness.Rank","Economy..GDP.per.Capita.","Health..Life.Expectancy.","Trust..Government.Corruption."]]
dataframe.loc[144:154,["Country","Happiness.Rank","Economy..GDP.per.Capita.","Health..Life.Expectancy.","Trust..Government.Corruption."]]
df=pd.read_csv("../input/2015.csv")
df.head(10)
df.loc[0:9,["Country","Region","Economy (GDP per Capita)","Health (Life Expectancy)","Freedom"]]
df.tail(10)
df=pd.read_csv("../input/2016.csv")
df.head(10)
df["Region"].unique()
print(len(df["Region"].unique()),"unique Region")