# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

import pandas_profiling as pp

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from plotly.offline import iplot

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
Data=pd.read_csv("/kaggle/input/online-retail-socgen/OnlineRetail.csv", encoding='iso-8859-1' )

print("Count of Rows, Columns: ",Data.shape)
Data.head()
Data.isnull().sum()
for k in Data.columns:

    print(k,Data[k].nunique())

    

Data.head()
bool_series = Data["InvoiceNo"].str.startswith("C", na = False) 

Invoice_Cancelled = Data[bool_series]

Invoice_Cancelled
Data['Date']=[item[0] for item in Data['InvoiceDate'].str.split()]

Data['Time']=[item[1] for item in Data['InvoiceDate'].str.split()]

Data['Month']=[item[1] for item in Data['Date'].str.split('-')]

Data['Year']=[item[2] for item in Data['Date'].str.split('-')]

Data['TotalCost']=Data['Quantity']*Data['UnitPrice']
Month={'1':'Jan' , '2':'Feb' , '3':'Mar', '4':'Apr' ,'5':'May' , '6':'Jun' ,

       '7':'Jul' , '8':'Aug' , '9':'Sep' , '10':'Oct', '11':'Nov' ,'12':'Dec',

       '01':'Jan' , '02':'Feb' , '03':'Mar', '04':'Apr' ,'05':'May' , '06':'Jun' ,

       '07':'Jul' , '08':'Aug' , '09':'Sep' }



Data=Data.replace({"Month": Month})

Data.head()
temp_df = Data.groupby(["Month"])["TotalCost"].agg(["size","mean"]).reset_index()

temp_df["Month"] = pd.to_datetime(temp_df.Month, format='%b', errors='coerce').dt.month

temp_df = temp_df.sort_values(by="Month")







trace = go.Scatter(

    x=temp_df['Month'],

    y=temp_df['size'],

    marker=dict(

        color="#1E90FF",

    ),

)



layout = go.Layout(

    title=go.layout.Title(

        text="Number of Sales - Month on Month",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="funding")
temp_df = Data.groupby(["Month"])["TotalCost"].agg(["mean"]).reset_index()

temp_df["Month"] = pd.to_datetime(temp_df.Month, format='%b', errors='coerce').dt.month

temp_df = temp_df.sort_values(by="Month")



trace = go.Scatter(

    x=temp_df['Month'],

    y=temp_df['mean'],

    marker=dict(

        color="#1E90FF",

    ),

)



layout = go.Layout(

    title=go.layout.Title(

        text="Average cost of Sales - Month on Month",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Retail")
temp_df
def horizontal_bar_chart(srs, color):

    trace = go.Bar(

        x=srs.values[::-1],

        y=srs.index[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_srs = Data['Country'].value_counts().head(7)

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Sales in Top 7 Country",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=800,

)



data = [horizontal_bar_chart(cnt_srs, "#1E90FF")]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Retail")

def horizontal_bar_chart(srs, color):

    trace = go.Bar(

        x=srs.values[::-1],

        y=srs.index[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_srs = Data['Country'].value_counts().tail(7)

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Sales in Least 7 Country",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=800,

)



data = [horizontal_bar_chart(cnt_srs, "#1E90FF")]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Retail")

temp_df = Data.groupby(["Country","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Country","Month", "Size", "Mean"]
temp_df = Data.groupby(["Country","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df
temp_df = Data.groupby(["Country","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Country","Month", "Size", "Mean"]

temp_df= temp_df.sort_values(by=['Mean'])



def horizontal_bar_chart(srs, color):

    trace = go.Bar(

        x=temp_df['Mean'].head(7),

        y=temp_df['Country'].head(7),

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_srs = temp_df['Mean'].head(7)

layout = go.Layout(

    title=go.layout.Title(

        text="Average Cost spend by Country",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=800,

)



data = [horizontal_bar_chart(cnt_srs, "#1E90FF")]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Retail")

temp_df.sort_values(by="Size", ascending=False).reset_index()[:100]
temp_df = Data.groupby(["Country","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Country","Month", "Size", "Mean"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

temp_df=temp_df.sort_values(by="Mean", ascending=False).reset_index()[:100]

temp_df=temp_df.head(50)

fig = px.scatter(temp_df, x="Month", y="Country", color="Country", size="Mean")

layout = go.Layout(

    title=go.layout.Title(

        text="Mean Purchase over Month by Country",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df = Data.groupby(["Country","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Country","Month", "Size", "Mean"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

temp_df=temp_df.sort_values(by="Size", ascending=False).reset_index()[:100]

#temp_df=tem

fig = px.scatter(temp_df, x="Month", y="Country", color="Country", size="Size")

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Purchase over Month by Country",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



inv_names = []

for invs in Data['Description']:

    for inv in str(invs).split():

        if inv != "":

            inv_names.append(inv.strip().lower().replace("'",""))

            

def plot_wordcloud(text, mask=None, max_words=40, max_font_size=80, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown', 'nan', ' nan'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    prefer_horizontal = 1.0,

                    max_font_size = max_font_size, 

                    min_font_size = 10,

                    random_state = 42,

                    #color_func = lambda *args, **kwargs: (140,0,0),

                    #color_func = color_map(),

                    colormap="Blues",

                    width=600, 

                    height=300,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        #image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_color), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size, 'color': 'blue',

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'blue', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  



plot_wordcloud(' '.join(inv_names), title="Most Sold Products")
Data['Description']=Data.groupby(["Country","UnitPrice","Date"])['Description'].transform(lambda x: x.fillna(x.mode()))

Data['Description']=Data['Description'].transform(lambda x: x.fillna("Others"))
temp_df = Data.groupby(["Description","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Description","Month", "Size", "Mean"]

temp_df= temp_df.sort_values(by=['Mean'])



temp_df=temp_df.sort_values(by="Size", ascending=False).reset_index()[:50]

#temp_df=tem

fig = px.scatter(temp_df, x="Month", y="Description", color="Description", size="Size")

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Purchase over Month by Description",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df = Data.groupby(["Description","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Description","Month", "Size", "Mean"]

temp_df= temp_df.sort_values(by=['Mean'])



temp_df=temp_df.sort_values(by="Size", ascending=False).reset_index().tail(50)

#temp_df=tem

fig = px.scatter(temp_df, x="Month", y="Description", color="Description", size="Size")

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Purchase over Month by Description",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df = Data.groupby(["Description","Month"])["TotalCost"].agg(["size", "mean"]).reset_index()

temp_df.columns = ["Description","Month", "Size", "Mean"]

temp_df= temp_df.sort_values(by=['Mean'])



temp_df=temp_df.sort_values(by="Size", ascending=False).reset_index().tail(50)

#temp_df=tem

fig = px.scatter(temp_df, x="Month", y="Description", color="Description", size="Size")

layout = go.Layout(

    title=go.layout.Title(

        text="Number of Purchase over Month by Description",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df = Data.groupby(["Country"])["TotalCost"].agg(["size", "sum"]).reset_index()

temp_df.columns = ["Country", "Size", "Sum"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

temp_df=temp_df.sort_values(by="Sum", ascending=False).reset_index().head(5)

temp_df=temp_df

fig = px.scatter(temp_df, x="Sum", y="Country", color="Country", size="Sum")

layout = go.Layout(

    title=go.layout.Title(

        text="Top 5 Countries Contributing towards revenue",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df = Data.groupby(["Country"])["TotalCost"].agg(["size", "sum"]).reset_index()

temp_df.columns = ["Country", "Size", "Sum"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

temp_df=temp_df.sort_values(by="Sum", ascending=False).reset_index().tail(5)

temp_df=temp_df

fig = px.scatter(temp_df, x="Sum", y="Country", color="Country", size="Sum")

layout = go.Layout(

    title=go.layout.Title(

        text="Last 5 Countries Contributing towards revenue",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=600,

    showlegend=False

)

fig.update_layout(layout)

fig.show()
temp_df
temp_df=Data[Data['Country'] == 'United Kingdom']

temp_df = Data.groupby(["Description"])["TotalCost"].agg(["size", "sum"]).reset_index()

temp_df.columns = ["Description", "Size", "Sum"]

#temp_df.Country = temp_df[temp_df.Country != 'United Kingdom']

temp_df=temp_df.sort_values(by="Sum", ascending=False).head(10)

temp_df=temp_df
def horizontal_bar_chart(srs, color):

    trace = go.Bar(

        x=temp_df['Sum'],

        y=temp_df['Description'],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_srs = temp_df['Sum']

layout = go.Layout(

    title=go.layout.Title(

        text="Top Sources Contributing to UK's Revenue",

        x=0.5

    ),

    font=dict(size=14),

    width=800,

    height=800,

)



data = [horizontal_bar_chart(cnt_srs, "#1E90FF")]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Retail")



df=Data[Data['Country'] == 'United Kingdom']

df = Data.groupby(["Description"])["TotalCost"].agg(["size", "sum"]).reset_index()

df.columns = ["Description", "Size", "Sum"]
print("Total  % of income of from these top 10 product in UK is",sum(temp_df['Sum'])/sum(df["Sum"])*100)
temp_df=Data[['Month','Description']]

temp_df["Month"] = pd.to_datetime(temp_df.Month, format='%b', errors='coerce').dt.month

temp_df = temp_df.sort_values(by="Month")
nunique()