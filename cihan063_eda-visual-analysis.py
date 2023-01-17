import numpy as np 

import pandas as pd 

from collections import Counter

from statistics import *
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import squarify

from wordcloud import WordCloud

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data2013=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2013')

data2014=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2014')

data2015=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2015')

data2016=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2016')

data2017=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2017')

data2018=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2018')

data2019=pd.read_excel(open('../input/impact-factor-of-top-1000-journals/Impact-Factor-Ratings.xlsx','rb'),sheet_name='2019')
data2013.columns = data2013.columns.str.replace(' ', '_')

data2013['Starting_year']=2010

data2013['Ending_year']=2013

data2013=data2013.rename(columns={'2010-13_Citations':'Citations',

                        '2010-13_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2014.columns = data2014.columns.str.replace(' ', '_')

data2014['Starting_year']=2011

data2014['Ending_year']=2014

data2014=data2014.rename(columns={'2011-14_Citations':'Citations',

                        '2011-14_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2015.columns = data2015.columns.str.replace(' ', '_')

data2015['Starting_year']=2012

data2015['Ending_year']=2015

data2015=data2015.rename(columns={'2012-15_Citations':'Citations',

                        '2012-15_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2016.columns = data2016.columns.str.replace(' ', '_')

data2016['Starting_year']=2013

data2016['Ending_year']=2016

data2016=data2016.rename(columns={'2013-16_Citations':'Citations',

                        '2013-16_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2017.columns = data2017.columns.str.replace(' ', '_')

data2017['Starting_year']=2014

data2017['Ending_year']=2017

data2017=data2017.rename(columns={'2014-17_Citations':'Citations',

                        '2014-17_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2018.columns = data2018.columns.str.replace(' ', '_')

data2018['Starting_year']=2015

data2018['Ending_year']=2018

data2018=data2018.rename(columns={'2015-18_Citations':'Citations',

                        '2015-18_Documents':'Documents',

                        '%_Cited':'%Cited'})



data2019.columns = data2019.columns.str.replace(' ', '_')

data2019['Starting_year']=2016

data2019['Ending_year']=2019

data2019=data2019.rename(columns={'2016-19_Citations':'Citations',

                        '2016-19_Documents':'Documents',

                        '%_Cited':'%Cited'})
frames = [data2013, data2014, data2015, data2016, data2017, data2018, data2019]

result_df = pd.concat(frames)
result_df['Category']=result_df['Highest_percentile'].apply(lambda x:x.split('\n')[2])
result_df['above2015'] = ['above2015'if i >=2015 else 'below2015'for i in result_df.Starting_year]
arr = result_df['%Cited'].unique()

print("Median of %Cited Values: ",np.median(arr))



result_df['Median_of_Cited'] = ['higher72.5'if i >=72.5 else 'below72.5'for i in result_df['%Cited']]
result_df.info()
result_df.head()
category_count = Counter(result_df.Category)

most_common_categories = category_count.most_common(15)

x,y = zip(*most_common_categories)

x,y= list(x), list(y)



plt.figure(figsize=(30,15))

ax=sns.barplot(x=y,y=x, palette=sns.cubehelix_palette(len(x)))

plt.xticks(rotation=45)

plt.title('Most Common 15 Categories ')
category_df=pd.DataFrame()

category_df['Category']=x

category_df['Values']=y



category_df
fig = plt.figure(figsize=(10,10))

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

ax.pie(category_df.Values, labels =category_df.Category ,autopct='%1.2f%%')

plt.show()
sns.swarmplot(x='above2015',y='Citations',hue='Median_of_Cited',data=result_df)

plt.show()
def drawThis(a,i):

    sns.countplot(x=result_df[a],  palette="Set2",ax=axs[i])

    axs[i].set_title(a, color='blue', fontsize=15)

    



a=['Median_of_Cited','above2015']

iterr= 0



numToPlot = len(a)

fig, axs =plt.subplots(ncols=numToPlot)

plt.subplots_adjust(right=2, wspace = 0.5)

for i in a:

    drawThis(i,iterr)

    iterr +=1
plt.style.use("classic")

sns.distplot(result_df['%Cited'], color='blue')

plt.xlabel("Cited")

plt.ylabel("Count")

plt.show()
sns.violinplot(x =result_df['Ending_year'], y =result_df['SJR'], hue =result_df['Median_of_Cited'], split = True)
sns.violinplot(x =result_df['Ending_year'], y =result_df['SNIP'], hue =result_df['Median_of_Cited'], split = True)
sns.stripplot(x ='Ending_year', y ='SJR', data = result_df,

              jitter = True, hue ='Median_of_Cited', dodge = True)
plt.style.use("fivethirtyeight")

plt.figure(figsize=(16, 6))

sns.kdeplot(result_df.loc[result_df['Category'] == 'General Medicine', '%Cited'], label = 'General Medicine',shade=True)

sns.kdeplot(result_df.loc[result_df['Category'] == 'Cardiology and Cardiovascular Medicine', '%Cited'], label = 'Cardiology and Cardiovascular Medicine',shade=True)

sns.kdeplot(result_df.loc[result_df['Category'] == 'Oncology', '%Cited'], label = 'Oncology',shade=True)

sns.kdeplot(result_df.loc[result_df['Category'] == 'Ecology, Evolution, Behavior and Systematics', '%Cited'], label = 'Ecology, Evolution, Behavior and Systematics',shade=True)
x2020=result_df.Category

plt.subplots(figsize=(10,19))

wordcloud = WordCloud(background_color = 'white',

                     width=512,

                     height=384).generate("".join(x2020))



plt.imshow(wordcloud)

plt.axis('off')