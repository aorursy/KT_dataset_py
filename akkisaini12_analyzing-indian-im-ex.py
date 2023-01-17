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

%matplotlib inline



import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
import cufflinks as cf

init_notebook_mode(connected=True)



cf.go_offline()
df_ex = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')

df_im = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
#To see how the data looks like we will check out its head by

df_ex.head()
df_im.head()
#we'll no just explore this data a little bit

df_ex.info()
df_im.info()
df_ex['HSCode'].nunique()
df_im['HSCode'].nunique()
df_ex.isnull().sum() #tells us the number of missing data 
df_im.isnull().sum()
df_ex['country'].nunique()
df_im['country'].nunique()
print("number of countries india is exporting to :{} " .format(df_ex['country'].nunique()))
print("number of countries india imports from:{}".format(df_im['country'].nunique()))
#now lets clean the data for unwanted entries and null values

def cleaning(data_df):

    data_df['country'] = data_df['country'].apply(lambda x : np.NaN if x == 'UNSPECIFIED' else x )

    data_df = data_df[data_df['value']!=0]

    data_df.dropna(inplace = True)

    return data_df
df_ex = cleaning(df_ex)

df_im = cleaning(df_im)
df_ex.isnull().sum()
df = pd.DataFrame(df_im['Commodity'].value_counts())
df.head(10) #to see the top 10 imported commodities in India
df3 = df_im.groupby('year').agg({'value': 'sum'})

df4 = df_ex.groupby('year').agg({'value': 'sum'})
df3['value_ex'] = df4.value

df3['deficit'] = df4.value - df3.value

df3
years = ['2010','2011','2012','2013','2014','2015','2016','2017','2018']

fig = go.Figure(data = [

    go.Bar(x = years, y= df3.value , name= 'import'),

    go.Bar(x=years, y =df3.deficit , name = 'deficit'),

    go.Bar(x=years, y = df3.value_ex , name= 'export')],

     layout= {'barmode':'group'})

fig.show()
#now lets suppose we want to check the countries from where the maximum import and export happen to and from India.
#we'll define some new variables



df5 = df_ex.groupby('country').agg({'value': 'sum'}) #for analysing exports

df5 = df5.sort_values(by='value',ascending = False)

df5 = df5.head(10)



df6 = df_im.groupby('country').agg({'value':'sum'}) #for analysing imports

df6 = df6.sort_values(by='value',ascending = False)

df6 = df6.head(10)

#you can also go ahead and check the head of it.
#this type of visualisation is done best by using pie charts. 

fig = go.Figure(data =[go.Pie(labels=df5.index,values=df5.value,title= 'Exported from India')])

fig.show()
fig = go.Figure(data =[go.Pie(labels=df6.index,values=df6.value,title= 'Imported to India')])

fig.show()
#now we know we export a lot to USA, by checking the difference between the thigs we used to 

#export to USA in 2010 and exports to USA in 2018, we can see how India has progressed.

df7 = df_ex[df_ex['country'] =='U S A']

df7.head()
df_7 = df7[df7['year']==2010].groupby('Commodity').agg({'value':'sum'})

df_7 = df_7.sort_values(by= 'value',ascending= False)

df_7 = df_7.head(10)
df_7a = df7[df7['year']==2018].groupby('Commodity').agg({'value':'sum'})

df_7a = df_7a.sort_values(by= 'value',ascending= False)

df_7a = df_7a.head(10)
plt.figure(figsize=(20,24))

ax1 = plt.subplot(211)

sns.barplot(df_7.value , df_7.index).set_title('exports to USA in 2010')



max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax1.get_yticklabels()]



ax1.set_yticklabels(new_labels)



ax2 = plt.subplot(212)

sns.barplot(df_7a.value,df_7a.index).set_title('exports to USA in 2018')

max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax2.get_yticklabels()]



ax2.set_yticklabels(new_labels)

plt.tight_layout()
#the graphs are here, i'll leave it to you to make your own interpretation, please do share in the comments section.
#now in the same way we can compare total exports and imports of 2010 with 2018,



#for imoorts of 2010

dff1 = df_im[df_im['year']==2010].groupby('Commodity').agg({'value':'sum'})

dff1 = dff1.sort_values(by = 'value', ascending =False)

dff1 = dff1.head(7)
#for imports of 2018

dff2 = df_im[df_im['year']==2018].groupby('Commodity').agg({'value':'sum'})

dff2 = dff2.sort_values(by= 'value' , ascending = False)

dff2 = dff2.head(7)
plt.figure(figsize=(20,24))

ax1 = plt.subplot(211)

sns.barplot(dff1.value,dff1.index).set_title('imports in 2010')

max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax1.get_yticklabels()]



ax1.set_yticklabels(new_labels)



ax2 = plt.subplot(212)

sns.barplot(dff2.value,dff2.index).set_title('imports in 2018')

max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax2.get_yticklabels()]



ax2.set_yticklabels(new_labels)

plt.tight_layout()
dff3 = df_ex[df_ex['year']==2010].groupby('Commodity').agg({'value':'sum'}) #for exports of 2010

dff3 = dff3.sort_values(by = 'value', ascending =False)

dff3 = dff3.head(7)
dff4 = df_ex[df_ex['year']==2018].groupby('Commodity').agg({'value':'sum'}) #for exports of 2018

dff4 = dff4.sort_values(by = 'value', ascending =False)

dff4 = dff4.head(7)
plt.figure(figsize=(20,24))

ax1 = plt.subplot(211)

sns.barplot(dff3.value,dff3.index).set_title('exports in 2010')

max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax1.get_yticklabels()]



ax1.set_yticklabels(new_labels)



ax2 = plt.subplot(212)

sns.barplot(dff4.value,dff4.index).set_title('exports in 2018')

max_chars = 20



new_labels = ['\n'.join(label._text[i:i + max_chars ] 

                        for i in range(0, len(label._text), max_chars ))

              for label in ax2.get_yticklabels()]



ax2.set_yticklabels(new_labels)

plt.tight_layout()
# 1. As seen from the graph India has nearly doubled it's exports in nuclear reactors, boilers etc and vehicles as well.

# 2. Feel free to make your own interpretation of the graphs, do let me know in comments,

#thank you so much. 