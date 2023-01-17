# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import seaborn as sns # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyplot

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly.graph_objs as go

import plotly

from plotly import tools

import plotly.express as px

from scipy.stats import boxcox

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 100)

from wordcloud import WordCloud, STOPWORDS 

df = pd.read_csv('/kaggle/input/carnegie-museum-of-art/cmoa.csv', delimiter=',')

df.head(1)

#looks good!
null_perc = df.isnull().sum()/len(df)*100

null_perc.sort_values(ascending = False).head(10)
df.head(10)
print(df['credit_line'].nunique())#Number of unique contribution avenues for the Museums collection. Far too many to graph

#my hunch is there is that a small percentage of the organizations do a vast majority of the work

plt.style.use('seaborn-darkgrid')#Visual Style

sns.set(rc={'figure.figsize':(15,8.27)})#Set Figure Size

ax=df['credit_line'].value_counts()[:10]

ax=ax.to_frame()

ax.iplot(kind='barh',xTitle = "Count", yTitle = 'Credit Line', title = 'Popular Credit line', color = 'orange')

plt.pyplot.show()
df.head(3)
Medium=df['medium'].str.split('on',1,expand=True) #Splitting the 'medium' column on the keyword 'on'

df['material']=Medium[1] #Reasiging everything after the keyword to a new column 'material'

df['art component']=Medium[0]#Reasiging everything before the keyword to a new column 'art component'

df.head(1)
df.head(3)
df['date_acquired']=pd.to_datetime(df['date_acquired'], infer_datetime_format=True) #Converting 'date_acquired' into something we can work with

df['year_acquired']=df['date_acquired'].dt.strftime('%Y') #Splitting off the 'year' for the 'date_acquired' column



df.head(1)
print(df['medium'].nunique())#Number of unique mediums in the Carnegie Museums collection. Far too many to graph

plt.style.use('seaborn-darkgrid')#Visual Style

sns.set(rc={'figure.figsize':(11.7,8.27)})

ax=df['medium'].value_counts()[:10].iplot(kind='barh',

                                          xTitle='Medium',

                                          yTitle='Pieces of art',

                                        title='Overall Top 10 Mediums')

plt.pyplot.show()
print(df['classification'].nunique()) #Returns number of unique classificaitons of art

plt.style.use('seaborn-darkgrid')#Visual Style

sns.set(rc={'figure.figsize':(11.7,8.27)})#Set Figure Size

ax=df['classification'].value_counts()[:10].iplot(kind='barh',

                                          xTitle='Classifications',

                                          yTitle='Pieces of art',

                                        title='Overall Top 10 Artistic Classifications')

plt.pyplot.show()
#Lets make a new DF that is home to our Top 10 overall 

#Mediums of art by acquisition number

df1=df[df['medium'].isin(['gelatin silver print',

                          'woodblock print on paper',

                          'lithograph on paper',

                          'etching','oil on canvas',

                          'engraving','woodcut on paper',

                          'porcelain','ink on linen',

                          'pencil on tracing paper'])]

df1.tail(1)
plt.style.use('seaborn-darkgrid')#Visual Style

sns.set_palette("tab20",5)#Color Scheme

df1.groupby(['year_acquired','art component']).count()['department'].unstack().iplot(ax=ax,

                                                                                    xTitle='Year',

                                                                                    yTitle='Number of Peices',

                                                                                    title='Popularity of Art Acquisition: Key Component of Art')

plt.pyplot.show()
plt.style.use('seaborn-darkgrid') #Visual Style

sns.set_palette("tab20",5) #Color Scheme

df1.groupby(['date_acquired','art component']).count()['department'].unstack().iplot(ax=ax,

                                                                                    xTitle='Year',

                                                                                    yTitle='Number of Peices',

                                                                                    title='Popularity of Artistic Elemetents\n Through Art Acquisition')

plt.pyplot.show()
plt.style.use('seaborn-darkgrid')#Visual Style

sns.set_palette("tab20",5)#Color Scheme

df.groupby(['date_acquired','department']).count()['classification'].unstack().iplot(ax=ax,

                                                                                    xTitle='Year',

                                                                                    yTitle='Number of Peices',

                                                                                    title='Popularity of Art Acquisition: Carnegie Art Departments')

plt.pyplot.show()
#Lets make a new DF that is home to our Top 10 overall 

#classifications of art by acquisition number

df3=df[df['classification'].isin(['print',

                                  'drawings and watercolors',

                                  'photographs',

                                  'Ceramics',

                                  'paintings',

                                  'Metals',

                                  'containers',

                                  'sculpture',

                                  'Glass',

                                  'Wood'])]

df3.tail(1)

plt.style.use('seaborn-darkgrid')#Visual Style

sns.set_palette("tab20",5)#Color Scheme

df3.groupby(['date_acquired','classification']).count()['department'].unstack().iplot(ax=ax,

                                                                                    xTitle='Year',

                                                                                    yTitle='Number of Peices',

                                                                                    title='Popularity of Art Acquisition: Art Classification')

plt.pyplot.show()
df.tail(1)
text = df['title'].values 

wordcloud = WordCloud().generate(str(text))



pyplot.figure(figsize = (8, 8), facecolor = 'white') 

pyplot.imshow(wordcloud)

pyplot.axis("off")

pyplot.tight_layout(pad = 0) 

pyplot.show()
text = df['medium'].values 

wordcloud = WordCloud().generate(str(text))



pyplot.figure(figsize = (8, 8), facecolor = 'white') 

pyplot.imshow(wordcloud)

pyplot.axis("off")

pyplot.tight_layout(pad = 0) 

pyplot.show()