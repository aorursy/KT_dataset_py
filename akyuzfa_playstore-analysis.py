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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#name of dataframe is ps (means playstore) 
ps = pd.read_csv('../input/googleplaystore.csv')

ps.info()
ps.head()
ps.drop_duplicates(subset='App', inplace= True)
print(len(ps))
#find the columns including "null"
null_columns=ps.columns[ps.isnull().any()]
print(ps[ps.isnull().any(axis=1)][null_columns].head())
ps_category = ps.groupby('Category')
print(ps_category.groups.keys())
print("Number of Category: " + str(ps_category.ngroups))
#there is a category named as "1.9". it should be deleted.
ps.drop(ps.loc[ps['Category']== '1.9'].index, inplace=True)
ps_type = ps.groupby('Type')
print(ps_type.groups.keys())
ps_rating = ps.groupby('Content Rating')
print(ps_rating.groups.keys())
ps_price= ps.groupby('Price')
print(ps_price.groups.keys())
print(ps_price.ngroups)
# Free apps in Education Category
ps[(ps['Category']=='EDUCATION') & (ps['Type'] == 'Free')]
#it is just a trial
for index,value in ps[['Category']][0:5].iterrows():
    print(index," : ",value)
ps.Rating.plot(kind = "hist",bins = 30, color = 'purple')
ps.Rating.mean()
