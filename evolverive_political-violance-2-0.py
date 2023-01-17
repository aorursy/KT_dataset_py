# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import squarify

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv('../input/greeceisraelsyriapolitical-events/2014-10-17-2019-10-17-Greece-Israel-Syria.csv')



# Any results you write to the current directory are saved as output.

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()



# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
#reducing dataset

print(df.columns)

df = df[['actor1', 'actor2','country', 'event_date', 'event_type', 'fatalities','latitude', 'location',

       'longitude', 'notes', 'source', 'year']]

#checking if ther's some empty field in percent

#quantity of null elemets division

round((df.isnull().sum()/df.shape[0])*100,2)
df.country.value_counts().head()

#how many conflicts in these countries
data_1 = df.country.value_counts().sort_values(ascending=False).head(10)

x=data_1.index

y= data_1.values



trace1 = go.Bar(

    x=x,

    y=y,

    text = y,

    textposition = 'auto',

    textfont = {'size':12,'color':'black'},

    marker=dict(

    color='SlateGray'),

    opacity=0.8,

    orientation ='v',

)



data = [trace1]



layout = go.Layout (

    yaxis = dict (

    title = 'Numbers of Conflitc'),

    

    xaxis = dict (

    title = 'Country'),

    

    title = 'Highest conflicts'

)

fig = go.Figure (data=data, layout = layout)

py.iplot(fig)
data_2 = df.groupby('country')['fatalities'].sum().sort_values(ascending=False).head(10)

x=data_2.index

y= data_2.values



trace1 = go.Bar(

    x=x,

    y=y,

    text = y,

    textposition = 'auto',

    textfont = {'size':12,'color':'white'},

    marker=dict(

    color='darkred'),

    opacity=0.8,

    orientation ='v',

)



data = [trace1]



layout = go.Layout (

    

    xaxis = dict (

    title = 'Countries Name'),

    

    title = 'Countries with Highest Fatalities'

)

fig = go.Figure (data=data, layout = layout)

py.iplot(fig)
import re

from nltk.corpus import stopwords

clean_1 = re.compile('[/(){}\[\]\|@,;]')

clean_2 = re.compile('[^0-9a-z #+_]')

def clean_text (text):

    text = text.lower()

    text = clean_1.sub(' ',text) # compile and replace those symbole by empty scpace

    text = clean_2.sub('',text)

    text_2 = [word.strip() for word in text.split() if not word in set(stopwords.words('english'))]

    new_text = ''

    for i in text_2:

        new_text +=i+' '

    text = new_text

    return text.strip()

from wordcloud import WordCloud



wc = WordCloud(max_font_size=50, width=600, height=300,colormap='Blues')

wc.generate(' '.join(df['notes'].values))



plt.figure(figsize=(15,8))

plt.imshow(wc,interpolation="bilinear")

plt.title("Most Used Words by New Agencies", fontsize=35)

plt.axis("off")

plt.show()
dfIsrael=df[df['country']=='Israel']

clean_1 = re.compile('[/(){}\[\]\|@,;]')

clean_2 = re.compile('[^0-9a-z #+_]')

def clean_text (text):

    text = text.lower()

    text = clean_1.sub(' ',text) # compile and replace those symbole by empty scpace

    text = clean_2.sub('',text)

    text_2 = [word.strip() for word in text.split() if not word in set(stopwords.words('english'))]

    new_text = ''

    for i in text_2:

        new_text +=i+' '

    text = new_text

    return text.strip()

from wordcloud import WordCloud



wc = WordCloud(max_font_size=50, width=600, height=300,colormap='Blues')

wc.generate(' '.join(dfIsrael['notes'].values))



plt.figure(figsize=(15,8))

plt.imshow(wc,interpolation="bilinear")

plt.title("Most Used Words by New Agencies", fontsize=35)

plt.axis("off")

plt.show()
dfGreece=df[df['country']=='Greece']

clean_1 = re.compile('[/(){}\[\]\|@,;]')

clean_2 = re.compile('[^0-9a-z #+_]')

def clean_text (text):

    text = text.lower()

    text = clean_1.sub(' ',text) # compile and replace those symbole by empty scpace

    text = clean_2.sub('',text)

    text_2 = [word.strip() for word in text.split() if not word in set(stopwords.words('english'))]

    new_text = ''

    for i in text_2:

        new_text +=i+' '

    text = new_text

    return text.strip()

from wordcloud import WordCloud



wc = WordCloud(max_font_size=50, width=600, height=300,colormap='Blues')

wc.generate(' '.join(dfGreece['notes'].values))



plt.figure(figsize=(15,8))

plt.imshow(wc,interpolation="bilinear")

plt.title("Most Used Words by New Agencies", fontsize=35)

plt.axis("off")

plt.show()


dfSyria=df[df['country']=='Syria']

# Lower all word in event_type

dfSyria.event_type = df.event_type.apply(lambda x: x.lower())

event_data = dfSyria.groupby('event_type').sum().reset_index()

# Create a new columns that count the numbers of counflicts 

d = dict(dfSyria.event_type.value_counts())

event_data['conflicts'] = event_data['event_type'].map(d)

# Sort the data by Fatalities

event_data.sort_values(by='fatalities', ascending=False,inplace=True)

#reduce the data to only 8 event type

event_data = event_data.head(8)





f, ax = plt.subplots(1,1,figsize = (10,10))

ax = event_data[['fatalities', 'conflicts']].plot(kind='barh',ax=ax,width=0.8,

              color=['dodgerblue', 'slategray'], fontsize=13);



ax.set_title("Causes of Conflicts in Syria",fontsize=20)

ax.set_ylabel("Event Type", fontsize=15)



ax.set_yticklabels(event_data.event_type.values)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+750, i.get_y()+.25, \

            str(int(round(((i.get_width())/1000))))+'k', fontsize=12, color='black')



# invert for largest on top 

ax.invert_yaxis()

sns.despine(bottom=True)

x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label

plt.legend(loc=(1.0,0.98),fontsize=13,ncol=2)

plt.show()
dfIsrael=df[df['country']=='Israel']

# Lower all word in event_type

dfIsrael.event_type = dfIsrael.event_type.apply(lambda x: x.lower())

event_data = dfIsrael.groupby('event_type').sum().reset_index()

# Create a new columns that count the numbers of counflicts 

d = dict(dfIsrael.event_type.value_counts())

event_data['conflicts'] = event_data['event_type'].map(d)

# Sort the data by Fatalities

event_data.sort_values(by='fatalities', ascending=False,inplace=True)

#reduce the data to only 8 event type

event_data = event_data.head(8)





f, ax = plt.subplots(1,1,figsize = (10,10))

ax = event_data[['fatalities', 'conflicts']].plot(kind='barh',ax=ax,width=0.8,

              color=['dodgerblue', 'slategray'], fontsize=13);



ax.set_title("Causes of Conflicts in Israel",fontsize=20)

ax.set_ylabel("Event Type", fontsize=15)



ax.set_yticklabels(event_data.event_type.values)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+750, i.get_y()+.25, \

            str(int(round(((i.get_width())/1000))))+'k', fontsize=12, color='black')



# invert for largest on top 

ax.invert_yaxis()

sns.despine(bottom=True)

x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label

plt.legend(loc=(1.0,0.98),fontsize=13,ncol=2)

plt.show()
dfGreece=df[df['country']=='Greece']

# Lower all word in event_type

dfGreece.event_type = dfGreece.event_type.apply(lambda x: x.lower())

event_data = dfGreece.groupby('event_type').sum().reset_index()

# Create a new columns that count the numbers of counflicts 

d = dict(dfGreece.event_type.value_counts())

event_data['conflicts'] = event_data['event_type'].map(d)

# Sort the data by Fatalities

event_data.sort_values(by='fatalities', ascending=False,inplace=True)

#reduce the data to only 8 event type

event_data = event_data.head(8)





f, ax = plt.subplots(1,1,figsize = (10,10))

ax = event_data[['fatalities', 'conflicts']].plot(kind='barh',ax=ax,width=0.8,

              color=['dodgerblue', 'slategray'], fontsize=13);



ax.set_title("Causes of Conflicts in Greece",fontsize=20)

ax.set_ylabel("Event Type", fontsize=15)



ax.set_yticklabels(event_data.event_type.values)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+750, i.get_y()+.25, \

            str(int(round(((i.get_width())/1000))))+'k', fontsize=12, color='black')



# invert for largest on top 

ax.invert_yaxis()

sns.despine(bottom=True)

x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label

plt.legend(loc=(1.0,0.98),fontsize=13,ncol=2)

plt.show()
import plotly

x=df.year.value_counts().sort_index().index



y= df.year.value_counts().sort_index().values

#Second Graph

x2=df.groupby('year')['fatalities'].sum().sort_index().index

y2= df.groupby('year')['fatalities'].sum().sort_index().values

plotly.__version__



from plotly.subplots import make_subplots

fig = go.Figure(layout_title_text="Number of Conflicts vs Fatalits")

fig = make_subplots(rows=1, cols=2)

fig.add_bar(x=x, y=y,  row=1, col=1 , name='fatalities')

fig.add_bar(x=x2,y=y2, row=1, col=1, name='conflicts')

fig.show()
sns.catplot(x='event_type' ,y='fatalities', data=dfSyria  , hue='year', height=6.5 , aspect=2.5 , kind='boxen')

plt.show();
plotPerColumnDistribution(dfSyria, 10, 5)


plotScatterMatrix(df, 20, 10)