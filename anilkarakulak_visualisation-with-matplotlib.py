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
movies_data=pd.read_csv("../input/tmdb_5000_movies.csv")
credits_data=pd.read_csv("../input/tmdb_5000_credits.csv")
movies_data=movies_data[movies_data["vote_count"]>1000]
movies_data.info()
credits_data.info()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(movies_data.corr(),annot=True, linewidths=.5, fmt= '.2f',ax=ax,square=True)
plt.show()
movies_data.columns
movies_data.plot(kind='scatter', x='budget', y='revenue',alpha =0.6,color = 'red',figsize=(10,10))
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.title('Budget - Revenue Correlation') 
plt.show()
movies_data[(movies_data["budget"]>100000000) & (movies_data["revenue"]>1000000000)]
threshold = sum(movies_data.revenue)/len(movies_data.revenue)
movies_data["revenues"] = ["high" if i > threshold else "low" for i in movies_data.revenue]
movies_data.loc[:20,["original_title","revenue","revenues"]]
print(movies_data['original_language'].value_counts())
movies_data.sort_values("vote_average",axis=0,ascending=False,inplace=True)

movies_data.boxplot(column="revenue",by="vote_average",figsize=(30,5))
plt.show()
new_data=movies_data.head()
pd.melt(frame=new_data, id_vars='original_title', value_vars=['vote_average'])
import warnings
warnings.filterwarnings("ignore")


date_list=movies_data["release_date"]
datetime_object=pd.to_datetime(date_list)
movies_data["date"]=datetime_object
movies_data=movies_data.set_index("date")
new_data=movies_data.head()
new_data
movies_data.resample("A").mean()
movies_data.vote_average.plot(color = 'r',label = 'vote_average',linewidth=1, alpha = 0.6,grid = True,linestyle = '-',figsize=(30,5))
