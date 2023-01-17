import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import types

import pandas as pd

from wordcloud import WordCloud, STOPWORDS

from PIL import Image
df_data_import = pd.read_csv('../input/india-trade-data/2018-2010_import.csv')
df_data_import.shape
df_data_export = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')
df_data_export.shape
df_data_import.head()
indexNames = df_data_import[ df_data_import['value'] == 0].index

df_data_import.drop(indexNames , inplace=True)
indexNames = df_data_export[ df_data_export['value'] == 0].index

df_data_export.drop(indexNames , inplace=True)
df_data_import.dropna(inplace=True)

df_data_export.dropna(inplace=True)
df_data_import.drop_duplicates(subset=None, keep='first',inplace=True)

df_data_export.drop_duplicates(subset=None, keep='first',inplace=True)



#duplicateRowsDF = df_data_import[df_data_import.duplicated()]
print(df_data_import.shape)

print(df_data_export.shape)
indexNames = df_data_export[ df_data_export['country'] == 'UNSPECIFIED'].index

df_data_export.drop(indexNames , inplace=True)



indexNames = df_data_import[ df_data_import['country'] == 'UNSPECIFIED'].index

df_data_import.drop(indexNames , inplace=True)
df_data_import.shape
df_data_export.shape
df_data_import.loc[(df_data_import['HSCode'] == 8) & (df_data_import['country'] == 'AFGHANISTAN TIS')  & (df_data_import['year'] == 2018)]
print(df_data_import.shape)

print(df_data_export.shape)
df_data_import[['value','year']].groupby(['year']).sum()
df_data_export[['value','year']].groupby(['year']).sum()
fig = plt.figure(figsize=(10, 10))

ax_import = fig.add_subplot(111)

ax_export = fig.add_subplot(111)

plt.ylabel('Million US$') # add y-label

plt.xlabel('Years') # add x-label

n = np.arange(9)

plt.xticks(n,['2010','2011','2012','2013','2014','2015','2016','2017','2018'])

plt.title('India\'s import\export from 2010 to 2018')

import_l,x=ax_import.plot(df_data_import[['value','year']].groupby(['year'],as_index=False).sum(),color='red')

export_l,y=ax_export.plot(df_data_export[['value','year']].groupby(['year'],as_index=False).sum(),color='green')

plt.legend([import_l, export_l], ['Import', 'Export'])
df_data_import_comwise=df_data_import[['HSCode','Commodity','value','year']].groupby(['HSCode','Commodity','year'],as_index=False).sum()

df_data_export_comwise=df_data_export[['HSCode','Commodity','value','year']].groupby(['HSCode','Commodity','year'],as_index=False).sum()

df_data_import_comwise.set_index('HSCode',inplace=True)

df_data_export_comwise.set_index('HSCode',inplace=True)
df_data_export_comwise[df_data_export_comwise['year']==2016].sort_values(by=['value'],ascending=False).head()
df_data_export_comwise[df_data_export_comwise['year']==2017].sort_values(by=['value'],ascending=False).head()
df_data_export_comwise[df_data_export_comwise['year']==2018].sort_values(by=['value'],ascending=False).head()
values_2016 = df_data_export_comwise.loc[(df_data_export_comwise.year == 2016)].sort_values('value',ascending=False)['value'].head()

values_2016.loc['Other'] = df_data_export_comwise.loc[(df_data_export_comwise.year == 2016)].sort_values('value',ascending=False)['value'][5:].sum()



values_2017 = df_data_export_comwise.loc[(df_data_import_comwise.year == 2017)].sort_values('value',ascending=False)['value'].head()

values_2017.loc['Other'] = df_data_export_comwise.loc[(df_data_export_comwise.year == 2017)].sort_values('value',ascending=False)['value'][5:].sum()



values_2018 = df_data_export_comwise.loc[(df_data_import_comwise.year == 2018)].sort_values('value',ascending=False)['value'].head()

values_2018.loc['Other'] = df_data_export_comwise.loc[(df_data_export_comwise.year == 2018)].sort_values('value',ascending=False)['value'][5:].sum()



print(values_2018)

print(df_data_export_comwise.loc[(df_data_export_comwise.year == 2018)].sort_values('value',ascending=False)['value'].sum())
fig = plt.figure(figsize=[20,20]) # create figure



ax0 = fig.add_subplot(2, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)

ax1 = fig.add_subplot(2, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

ax2 = fig.add_subplot(2, 2, 3) # add subplot 1 (1 row, 2 columns, first plot)



labels = values_2016.index

sizes = values_2016

explode = (0.1, 0,0, 0, 0,0)





# Subplot 1: 2016 Pie plot

ax0.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax0.set_title ('2016 Export Data Commodity Wise')



labels = values_2017.index

sizes = values_2017

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title ('2017 Export Data Commodity Wise')



labels = values_2018.index

sizes = values_2018

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title ('2018 Export Data Commodity Wise')



plt.show()
df_data_import_comwise[df_data_export_comwise['year']==2016].sort_values(by=['value'],ascending=False).head()
df_data_import_comwise[df_data_export_comwise['year']==2017].sort_values(by=['value'],ascending=False).head()
df_data_import_comwise[df_data_export_comwise['year']==2018].sort_values(by=['value'],ascending=False).head()
values_2016 = df_data_import_comwise.loc[(df_data_import_comwise.year == 2016)].sort_values('value',ascending=False)['value'].head()

values_2016.loc['Other'] = df_data_import_comwise.loc[(df_data_import_comwise.year == 2016)].sort_values('value',ascending=False)['value'][5:].sum()



values_2017 = df_data_import_comwise.loc[(df_data_import_comwise.year == 2017)].sort_values('value',ascending=False)['value'].head()

values_2017.loc['Other'] = df_data_import_comwise.loc[(df_data_import_comwise.year == 2017)].sort_values('value',ascending=False)['value'][5:].sum()



values_2018 = df_data_import_comwise.loc[(df_data_import_comwise.year == 2018)].sort_values('value',ascending=False)['value'].head()

values_2018.loc['Other'] = df_data_import_comwise.loc[(df_data_import_comwise.year == 2018)].sort_values('value',ascending=False)['value'][5:].sum()

fig = plt.figure(figsize=[20,20]) # create figure



ax0 = fig.add_subplot(2, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)

ax1 = fig.add_subplot(2, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

ax2 = fig.add_subplot(2, 2, 3) # add subplot 1 (1 row, 2 columns, first plot)



labels = values_2016.index

sizes = values_2016

explode = (0.1, 0,0, 0, 0,0)





# Subplot 1: 2016 Pie plot

ax0.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax0.set_title ('2016 Import Data Commodity Wise')



labels = values_2017.index

sizes = values_2017

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title ('2017 Import Data Commodity Wise')



labels = values_2018.index

sizes = values_2018

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title ('2018 Import Data Commodity Wise')



plt.show()
df_import_pivot = df_data_import.pivot_table(index='year',columns=['HSCode','Commodity','country'],aggfunc=sum, fill_value=0).T.reset_index()

df_import_pivot.set_index('HSCode',inplace=True)





df_export_pivot = df_data_export.pivot_table(index='year',columns=['HSCode','Commodity','country'],aggfunc=sum, fill_value=0).T.reset_index()

df_export_pivot.set_index('HSCode',inplace=True)

df_import_pivot.sort_values(by=[2016],ascending=False).head()
df_export_pivot.sort_values(by=[2016],ascending=False).head()
df_import_pivot['diff_2017_2016'] = df_import_pivot[2017] - df_import_pivot[2016]

df_import_pivot['diff_2018_2017'] = df_import_pivot[2018] - df_import_pivot[2017]
df_import_pivot.sort_values(by=['diff_2017_2016'],ascending=False).head()
df_import_pivot[['Commodity','country',2010,2011,2012,2013,2014,2015,2016,2017,2018,'diff_2018_2017']].sort_values(by=['diff_2018_2017'],ascending=False).head()
print(df_import_pivot.sort_values(by=['diff_2017_2016'],ascending=False)['diff_2017_2016'].head().sum())

print(df_import_pivot.sort_values(by=['diff_2017_2016'],ascending=False)['diff_2017_2016'].sum())



print(df_import_pivot.sort_values(by=['diff_2018_2017'],ascending=False)['diff_2018_2017'].head().sum())

print(df_import_pivot.sort_values(by=['diff_2018_2017'],ascending=False)['diff_2018_2017'].sum())
df_export_pivot['diff_2017_2016'] = df_export_pivot[2017] - df_export_pivot[2016]

df_export_pivot['diff_2018_2017'] = df_export_pivot[2018] - df_export_pivot[2017]
df_export_pivot.sort_values(by=['diff_2017_2016'],ascending=False).head()
df_export_pivot.sort_values(by=['diff_2018_2017'],ascending=False).head()
df_data_import_countrywise=df_data_import[['country','value','year']].groupby(['country','year'],as_index=False).sum()

df_data_export_countrywise=df_data_export[['country','value','year']].groupby(['country','year'],as_index=False).sum()

df_data_import_countrywise.set_index('country',inplace=True)

df_data_export_countrywise.set_index('country',inplace=True)
df_data_import_countrywise[df_data_import_countrywise['year']==2016].sort_values(by=['value'],ascending=False).head()
df_data_import_countrywise[df_data_import_countrywise['year']==2017].sort_values(by=['value'],ascending=False).head()
df_data_import_countrywise[df_data_import_countrywise['year']==2018].sort_values(by=['value'],ascending=False).head()
values_2016 = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2016)].sort_values('value',ascending=False)['value'].head()

values_2016.loc['Other'] = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2016)].sort_values('value',ascending=False)['value'][5:].sum()



values_2017 = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2017)].sort_values('value',ascending=False)['value'].head()

values_2017.loc['Other'] = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2017)].sort_values('value',ascending=False)['value'][5:].sum()



values_2018 = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2018)].sort_values('value',ascending=False)['value'].head()

values_2018.loc['Other'] = df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2018)].sort_values('value',ascending=False)['value'][5:].sum()



print(values_2018)

print(df_data_import_countrywise.loc[(df_data_import_countrywise.year == 2018)].sort_values('value',ascending=False)['value'].sum())
fig = plt.figure(figsize=[20,20]) # create figure



ax0 = fig.add_subplot(2, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)

ax1 = fig.add_subplot(2, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

ax2 = fig.add_subplot(2, 2, 3) # add subplot 1 (1 row, 2 columns, first plot)



labels = values_2016.index

sizes = values_2016

explode = (0.1, 0,0, 0, 0,0)





# Subplot 1: 2016 Pie plot

ax0.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax0.set_title ('2016 Import Data Country Wise')



labels = values_2017.index

sizes = values_2017

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title ('2017 Import Data Country Wise')



labels = values_2018.index

sizes = values_2018

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title ('2018 Import Data Country Wise')



plt.show()
df_data_export_countrywise[df_data_export_countrywise['year']==2016].sort_values(by=['value'],ascending=False).head()
df_data_export_countrywise[df_data_export_countrywise['year']==2017].sort_values(by=['value'],ascending=False).head()
df_data_export_countrywise[df_data_export_countrywise['year']==2018].sort_values(by=['value'],ascending=False).head()
values_2016 = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2016)].sort_values('value',ascending=False)['value'].head()

values_2016.loc['Other'] = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2016)].sort_values('value',ascending=False)['value'][5:].sum()



values_2017 = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2017)].sort_values('value',ascending=False)['value'].head()

values_2017.loc['Other'] = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2017)].sort_values('value',ascending=False)['value'][5:].sum()



values_2018 = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2018)].sort_values('value',ascending=False)['value'].head()

values_2018.loc['Other'] = df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2018)].sort_values('value',ascending=False)['value'][5:].sum()



print(values_2018)

print(df_data_export_countrywise.loc[(df_data_export_countrywise.year == 2018)].sort_values('value',ascending=False)['value'].sum())
fig = plt.figure(figsize=[20,20]) # create figure



ax0 = fig.add_subplot(2, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)

ax1 = fig.add_subplot(2, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

ax2 = fig.add_subplot(2, 2, 3) # add subplot 1 (1 row, 2 columns, first plot)



labels = values_2016.index

sizes = values_2016

explode = (0.1, 0,0, 0, 0,0)





# Subplot 1: 2016 Pie plot

ax0.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax0.set_title ('2016 Export Data Country Wise')



labels = values_2017.index

sizes = values_2017

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.set_title ('2017 Export Data Country Wise')



labels = values_2018.index

sizes = values_2018

explode = (0.1, 0,0, 0, 0,0)



# Subplot 1: 2016 Pie plot

ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.set_title ('2018 Export Data Country Wise')



plt.show()
df_export_wrod_cloud=pd.DataFrame(df_data_export_countrywise.groupby(df_data_export_countrywise.index)['value'].sum())
df_export_wrod_cloud['value'].sum()
df_export_wrod_cloud['percentage']=(df_export_wrod_cloud['value']/df_export_wrod_cloud['value'].sum())*10000
s_export_world_cloud = ''

for index, row in df_export_wrod_cloud.iterrows():

     s_export_world_cloud = s_export_world_cloud + (index.replace(' ','') + ',') * row[1].astype(int) + ','

india_mask = np.array(Image.open('india-png--1200.png'))

# instantiate a word cloud object

export_wc = WordCloud(background_color='white', max_words=2000, mask=india_mask)



# generate the word cloud

export_wc.generate(s_export_world_cloud)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(export_wc, interpolation='bilinear')

plt.axis('off')

plt.show()

df_import_wrod_cloud=pd.DataFrame(df_data_import_countrywise.groupby(df_data_import_countrywise.index)['value'].sum())

#.apply(lambda x:100 * x / float(x.sum()))
df_import_wrod_cloud['percentage']=(df_import_wrod_cloud['value']/df_import_wrod_cloud['value'].sum())*10000
s_import_world_cloud = ''

for index, row in df_import_wrod_cloud.iterrows():

     s_import_world_cloud = s_import_world_cloud + (index.replace(' ','') + ',') * row[1].astype(int) + ','

india_mask = np.array(Image.open('india-png--1200.png'))

# instantiate a word cloud object

import_wc = WordCloud(background_color='white', max_words=2000, mask=india_mask)



# generate the word cloud

import_wc.generate(s_import_world_cloud)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(14) # set width

fig.set_figheight(18) # set height



plt.imshow(import_wc, interpolation='bilinear')

plt.axis('off')

plt.show()