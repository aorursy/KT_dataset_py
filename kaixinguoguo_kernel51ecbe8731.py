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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn")
df=pd.read_csv("../input/train-data-year/train_data_9.csv")
sns.boxplot(data=df,x='has_homepage',y='revenue',showmeans=True)
# 第一张图

plt.subplot(2,3,1)

sns.countplot(y="has_homepage",data=df)



# 第二张图

plt.subplot(2,3,2)

sns.countplot(y="belongs_to_collection",data=df)



# 第三张图

plt.subplot(2,3,3)

sns.countplot(y="has_tagline",data=df)



# 第四张图

plt.subplot(2,3,4)

sns.boxplot(data=df,x='has_homepage',y='revenue',showmeans=True)



# 第五张图

plt.subplot(2,3,5)

sns.boxplot(data=df,x='belongs_to_collection',y='revenue',showmeans=True)



#第六张图

plt.subplot(2,3,6)

sns.boxplot(data=df,x='has_tagline',y='revenue',showmeans=True)
### 1.随年份变化，电影平均预算和收入变化情况
#每年电影预算、电影收入的均值

year_budget_mean=df.budget.groupby(df.year).mean()

year_revenue_mean=df.revenue.groupby(df.year).mean()

plt.plot(year_budget_mean.index,year_budget_mean,label="budget_mean")

plt.plot(year_budget_mean.index,year_revenue_mean,label="revenue_mean")

plt.xticks(range(1920,2020,8))

plt.ylabel('average')

plt.xlabel('year')

plt.legend(loc='best')
#1980-2020年电影总数、电影总收入

year_film_total=df.id.groupby(df.year).count()

year_revenue_total=df.revenue.groupby(df.year).sum()

plt.plot(year_budget_mean.index,year_film_total,label="film_count")

plt.xticks(range(1920,2019,8))
plt.plot(year_budget_mean.index,year_revenue_total,label="year_revenue_total")
month_film=df.id.groupby(df.month).count()

day_film=df.id.groupby(df.day).count()

quarter_film=df.id.groupby(df.quarter).count()

dayofweek_film=df.id.groupby(df.dayofweek).count()
pic1=plt.subplot(2,2,1)

pic2=plt.subplot(2,2,2)

pic3=plt.subplot(2,2,3)

pic4=plt.subplot(2,2,4)

pic1.plot(month_film.index,month_film,label="month_total")

pic2.plot(day_film.index,day_film,label='day_total')

pic3.bar(quarter_film.index,quarter_film,label="quarter_total")

pic4.bar(dayofweek_film.index,dayofweek_film,label="dayofweek_total")

plt.legend()

plt.show()
## 每个月，每个季度，每天，每周的平均收入

month_revenue=df.revenue.groupby(df.month).mean()

day_revenue=df.revenue.groupby(df.day).mean()

quarter_revenue=df.revenue.groupby(df.quarter).mean()

dayofweek_revenue=df.revenue.groupby(df.dayofweek).mean()
pic1=plt.subplot(2,2,1)

pic2=plt.subplot(2,2,2)

pic3=plt.subplot(2,2,3)

pic4=plt.subplot(2,2,4)

pic1.plot(month_revenue.index,month_revenue,label="month_total")

pic2.plot(day_revenue.index,day_revenue,label='day_total')

pic3.bar(quarter_revenue.index,quarter_revenue,label="quarter_total")

pic4.bar(dayofweek_revenue.index,dayofweek_revenue,label="dayofweek_total")

plt.legend()

plt.show()
#双变量关系图

sns.jointplot(data=df,x='runtime',y='revenue')
filmtype=pd.read_csv("../input/filmtype/type_results.csv")
plt.figure(figsize=(20,8))

plt.bar(filmtype["type"],filmtype["times"])
filmtype_list={}

for i in range(filmtype.shape[0]):

    filmtype_list[filmtype.iloc[i,0]]=filmtype.iloc[i,1]
from wordcloud import WordCloud

fig,ax=plt.subplots(figsize=(12,6))

w = WordCloud( \

    width = 1000, height = 700,\

    background_color = "white",

    collocations=False

    ).fit_words(filmtype_list)

plt.imshow(w)

plt.grid(False)
pic1=plt.subplot(2,1,1)

pic2=plt.subplot(2,1,2)

pic1.scatter(df.cast_num,df.revenue)

pic2.scatter(df.crew_num,df.revenue)