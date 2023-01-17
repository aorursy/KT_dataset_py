# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
TedTalks=pd.read_csv('../input/ted_main.csv')
TedTalks.columns
TedTalks.shape
TedTalks.dtypes
TedIntColumns=TedTalks.select_dtypes(include=['int64'])

TedIntColumns.head()
TedIntColumns.corr()
TedIntColumns.boxplot()
sns.heatmap(TedIntColumns.corr())
TedIntColumns.describe()
TedIntColumns.boxplot('comments')
TedTalks.sort_values('views',ascending=False).head(10)
TedTalks[['title','main_speaker']][TedTalks.views==max(TedTalks.views)]
TedTalks[['title','main_speaker','views']][TedTalks.title.str.contains('school')].sort_values('views',ascending=False)
TedTalks.loc[TedTalks['main_speaker']=='Ken Robinson'].sort_values('views',ascending=False)
TedTalks[['title','main_speaker','views']][TedTalks['title'].str.contains('education')]
TedTalks['FirstName']=TedTalks['main_speaker'].apply(lambda x:x.split()[0])

TedTalks['FirstName'].head()
TedTalks.groupby('main_speaker').views.sum().nlargest(10).plot.bar()
TedTalks.groupby('main_speaker').views.mean().nlargest(10).plot.bar()
TedTalks.groupby('main_speaker').views.count().nlargest(10).plot.bar()
TedTalks.columns
TedTalks[['title','main_speaker','views','comments']].sort_values('comments',ascending=False).head(10)
TedTalks[['title','main_speaker','views','comments','duration']].sort_values('duration',ascending=False).head(10)
import datetime

TedTalks['film_date']=pd.to_datetime(TedTalks['film_date'],unit='s')

TedTalks['published_date']=pd.to_datetime(TedTalks['published_date'],unit='s')
TedTalks.groupby(TedTalks.published_date.dt.year).title.count().plot.bar()
TedTalks['year']=TedTalks.published_date.dt.year

TedTalks['month']=TedTalks.published_date.dt.month

TedTalks.groupby(['year','month']).title.count().plot.line()
Ted_month=TedTalks.groupby(['year','month']).title.count().reset_index(name='talks')

Ted_month.head()

#Ted_month.fillna(0,inplace=True)

Ted_month=Ted_month.pivot('year','month','talks')

Ted_month.fillna(0,inplace=True)

Ted_month
sns.heatmap(Ted_month)
import ast

TedTalks['ratings'] = TedTalks['ratings'].apply(lambda x: ast.literal_eval(x))

TedTalks['funny'] = TedTalks['ratings'].apply(lambda x: x[0]['count'])

TedTalks['jawdrop'] = TedTalks['ratings'].apply(lambda x: x[-3]['count'])

TedTalks['beautiful'] = TedTalks['ratings'].apply(lambda x: x[3]['count'])

TedTalks['confusing'] = TedTalks['ratings'].apply(lambda x: x[2]['count'])

TedTalks.head()







    

TedTalks[['title','main_speaker','funny']].sort_values('funny',ascending=False).head(10).plot.bar(x='title')
TedTalks[['title','main_speaker','jawdrop']].sort_values('jawdrop',ascending=False).head(10).plot.bar(x='title')
TedTalks[['title','main_speaker','beautiful']].sort_values('beautiful',ascending=False).head(10).plot.bar(x='title')
TedTalks[['title','main_speaker','confusing']].sort_values('confusing',ascending=False).head(10).plot.bar(x='title')
TedTalks[TedTalks['tags'].str.contains('children')]