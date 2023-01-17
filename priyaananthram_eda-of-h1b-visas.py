# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.import pandas as pd

h1b=pd.read_csv("../input/h1b_kaggle.csv")

h1b.SOC_NAME=h1b.SOC_NAME.apply(lambda x:str(x).lower())

h1b.head()
%matplotlib inline

import ggplot

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,5)

(h1b.CASE_STATUS.value_counts(normalize=True)*100).plot(kind='barh',title='H1B Petitions by Case Status')
plt.rcParams["figure.figsize"] = (20,10)



pd.crosstab(h1b.YEAR,h1b.CASE_STATUS).plot(title='H1Bs by year')
h1b_minus_certified=h1b[h1b.CASE_STATUS!='CERTIFIED']

pd.crosstab(h1b_minus_certified.YEAR,h1b_minus_certified.CASE_STATUS).plot(title='H1Bs by year')


plt.rcParams["figure.figsize"] = (20,10)

h1b.groupby('EMPLOYER_NAME').CASE_STATUS.count().nlargest(20).plot(kind='barh',title='Top 20 employers filing H1Bs')
top_20=h1b.groupby('EMPLOYER_NAME').CASE_STATUS.count().nlargest(20).index.tolist()

top_20_df=h1b.loc[h1b.EMPLOYER_NAME.isin(top_20)]

top_20_df.groupby('JOB_TITLE').EMPLOYER_NAME.count().nlargest(10).plot(kind='barh',title='Job title of top 20 h1b companies')
top_20_df.groupby('SOC_NAME').EMPLOYER_NAME.count().nlargest(10).plot(kind='barh',title='Occupation of the top20 h1b companies')
pd.crosstab(h1b.CASE_STATUS,h1b.FULL_TIME_POSITION).plot(kind='barh')
pd.crosstab(h1b.YEAR,h1b.FULL_TIME_POSITION).plot(title='Full time position over time')
common_jobs=h1b.groupby('JOB_TITLE').EMPLOYER_NAME.count().sort_values(ascending=False).index[0:20]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))





top_20_df_common_jobs=top_20_df[top_20_df.JOB_TITLE.isin(common_jobs)]

top_20_df_common_jobs=top_20_df_common_jobs[top_20_df_common_jobs.PREVAILING_WAGE<=top_20_df_common_jobs.PREVAILING_WAGE.quantile(0.99)]

top_20_df_common_jobs.boxplot(column='PREVAILING_WAGE',by='JOB_TITLE',vert=False,ax=axes[0])

h1b_common_jobs=h1b[h1b.JOB_TITLE.isin(common_jobs)]

h1b_common_jobs=h1b_common_jobs[h1b_common_jobs.PREVAILING_WAGE<=h1b_common_jobs.PREVAILING_WAGE.quantile(0.99)]



h1b_common_jobs.boxplot(column='PREVAILING_WAGE',by='JOB_TITLE',vert=False,ax=axes[1])

plt.yticks([])
top_20_df_common_jobs.boxplot(column='PREVAILING_WAGE',by='EMPLOYER_NAME',vert=False)
one=h1b.groupby('YEAR').PREVAILING_WAGE.median()

two=top_20_df.groupby('YEAR').PREVAILING_WAGE.median()

one=pd.DataFrame(one)

one['PREVAILING_WAGE_TOP20']=two



one.plot(title='Prevailing wage of the top20 verus everyone else')
df=h1b.groupby(['lon','lat'],as_index=False).PREVAILING_WAGE.median()

bins=[10000,40000,50000,60000,70000,90000,190000]



categories = pd.cut(df.PREVAILING_WAGE, bins)

df['categories']=categories

df.categories=df.categories.astype('str')

vvlow=df[df.categories=='(10000, 40000]']



vlow=df[df.categories=='(40000, 50000]']

low=df[df.categories=='(50000, 60000]']

medium=df[df.categories=='(60000, 70000]']



high=df[df.categories=='(70000, 90000]']

vhigh=df[df.categories=='(90000, 190000]']

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (25,10)



m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64,

  urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,

  lon_0=-95, resolution='h', area_thresh=10000)



# draw the coastlines of continental area

m.drawcoastlines()

m.drawstates()

# draw country boundaries

m.drawcountries(linewidth=2)

colors = ['blue','cyan','green','yellow','orange','red']

x, y = m(list(vvlow.lon.astype(float)), list(vvlow.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[0],label='Low')



# draw states boundaries (America only)

x, y = m(list(vlow.lon.astype(float)), list(vlow.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[1],label='Low')



x, y = m(list(low.lon.astype(float)), list(low.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[2],label='Low')





x, y = m(list(medium.lon.astype(float)), list(medium.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[3],label='')





x, y = m(list(high.lon.astype(float)), list(high.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[4],label='high')





x, y = m(list(vhigh.lon.astype(float)), list(vhigh.lat.astype(float)))

plot1=m.plot(x, y, 'go', markersize = 4, alpha = 0.8, color = colors[5],label='v high')

plt.legend()

plt.show()
