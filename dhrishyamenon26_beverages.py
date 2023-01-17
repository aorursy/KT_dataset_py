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
import pandas as pd
df=pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")
df
df.head(10)
df.tail()
df.iloc[:4,:3]
df.iloc[4:,3:]
x=df.iloc[3:8,1:6]
x
df.loc[:,"Calories"]
df.columns
df.loc[:4,"Calories"]
a=df.loc[(df['Beverage_prep']=='Brewed Coffee')&(df['Calories'])<10]
a