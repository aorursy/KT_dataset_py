#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSDghtT2lK3QZj_Shs88F3HGKmaZ2_HqvciNXWmpNu6r8rlrB_4&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTyaViHTugIlu6zwS1tMIOtDF3bp7124iVIEi1znwkLT3HzqoCk&usqp=CAU',width=400,height=400)
df = pd.read_csv("../input/uncover/UNCOVER/hifld/hifld/urgent-care-facilities.csv")

df.head().style.background_gradient(cmap='summer')
data = df.loc[:, ["city", "st_vendor", "geometry"]].copy()
# How many lands of each type are there?

df.st_vendor.value_counts()
# Select lands that fall under the "WILD FOREST" or "WILDERNESS" category Alexis mini-course

vendors = df.loc[df.st_vendor.isin(['NAVTEQ', 'TGS'])].copy()

vendors.head()
vendors.plot()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# View the first five entries in the "geometry" column

vendors.geometry.head()
df["fips"].plot.hist()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQjEAlfNb-Mgb5yPowiyEAUrx6MPki0mU928S4LM4cx8Xw8EFe9&usqp=CAU',width=400,height=400)
plt.style.use('dark_background')

sns.jointplot(df['naicscode'],df['fips'],data=df,kind='scatter')
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='naicscode',y='fips',data=df)
plt.style.use('dark_background')

sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="naicscode", y="fips", data=df)
plt.style.use('dark_background')

#sns.set(style="whitegrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

ax = sns.violinplot(x="naicscode", y="fips", data=df, inner=None)

ax = sns.swarmplot(x="naicscode", y="fips", data=df,color="white", edgecolor="black")
df.plot.area(y=['naicscode','fips','zip','id'],alpha=0.4,figsize=(12, 6));
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='YlOrRd_r')

plt.show()
corr = df.corr(method='pearson')

sns.heatmap(corr)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.city)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/cusersmarildownloadsventilatorcsv/ventilator.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df1.dataframeName = 'ventilator.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head()
cnt_srs = df1['quantity'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Ventilators Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="quantity")
fig = px.pie( values=df1.groupby(['quantity']).size().values,names=df1.groupby(['quantity']).size().index)

fig.update_layout(

    title = "Ventilators in São Paulo State",

    font=dict(

        family="Arial, monospace",

        size=10,

        color="#7f7f7f"

    )

    )   

    

py.iplot(fig)
cat = []

num = []

for col in df1.columns:

    if df1[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.figure(figsize=(18,9))

sns.factorplot(x=col,y='quantity',data=df1)

plt.tight_layout()

plt.show()
#Code from Prashant Banerjee.  https://www.kaggle.com/prashant111/extensive-analysis-and-visualization-fifa19
labels = df1['quantity'].value_counts().index

size = df1['quantity'].value_counts()

colors=['cyan','pink']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Quantity of Ventilators', fontsize = 20)

plt.legend()

plt.show()
df2 = pd.read_csv('../input/uncover/UNCOVER/ihme/2020_03_30/Hospitalization_all_locs.csv', encoding='ISO-8859-2')

df2.head()
#Let's visualise the evolution of results

hospital = df2.groupby('location').sum()[['allbed_mean', 'ICUbed_mean', 'InvVen_mean']]

#evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

#evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

hospital.head()
#plt.style.use('dark_background')

#What about the evolution of China Diagnosed Worldometer rate ?

plt.figure(figsize=(20,7))

plt.plot(hospital['InvVen_mean'], label='Ventilators mean')

plt.legend()

#plt.grid()

plt.title('Ventilators by US States')

plt.xticks(hospital.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRADJb9nItbXr8ly6TkktqqTA2u3JbEfUBerJXsfluyh0kzx6FM&usqp=CAU',width=400,height=400)