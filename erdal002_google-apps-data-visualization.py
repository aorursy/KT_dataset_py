# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as pltimport 
import matplotlib.pyplot as plt # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
google=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
google.head()
df=google.copy()
#I rename columns that could cause problem when making visualization

df.rename(columns={"Content Rating":"Content_Rating","Last Updated":"Last_Updated","Android Ver":"Androidver","Current Ver":"Currentver"},inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.head()
df.info()
#I want to use only year for visualization

year=[]

for i in df.Last_Updated:
    year.append(i.split(","))
    
  
yr_list=[]

for i in year:
    yr_list.append(i[1])
year_list=[]
for i in yr_list:
    i.split(" ")
    year_list.append(i)
df["Year"]=year_list
df.head()
df.Year=df.Year.astype(int)
df.Reviews=df.Reviews.astype(int)
Data=pd.DataFrame({"Genres":df.Genres.value_counts().index,"Number_of_Apps":df.Genres.value_counts().values})
Data=Data[Data["Number_of_Apps"]>=10]
fig = px.bar(Data, y="Number_of_Apps", x="Genres", text='Genres',color="Number_of_Apps")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text=" Number of Apps and Their Genres ")
fig.show()
Data=pd.DataFrame({"Android_ver":df.Androidver.value_counts().index,"Number_of_app":df.Androidver.value_counts().values})
plt.figure(figsize=(15,15))
sns.barplot(x="Android_ver",y="Number_of_app",data=Data,errwidth=0)
plt.xticks(rotation=90)
plt.xlabel("Android Versions of Apps")
plt.ylabel("Number of Apps")
plt.title("Number of Apps by Android Version")
plt.figure(figsize=(10,20))
sns.barplot(x="Type",y="Rating",hue="Category",data=df,errwidth=0)
Data=pd.DataFrame({"Size":df.Androidver.value_counts().index,"Number_of_app":df.Androidver.value_counts().values})
year=[2014,2015,2016,2017,2018]


for i in year:
    
    x=df[df["Year"]==i]

    plt.figure(figsize=(5,5))
    sns.barplot(x=x.Size.value_counts().index[:20] ,y=x.Size.value_counts().values[:20],data=x,errwidth=0)
    plt.xticks(rotation=90)
    plt.xlabel("Size of Apps")
    plt.ylabel("Number of Apps")
    plt.title("Number of Apps by Size in {}".format(i))
Data=pd.DataFrame(df.groupby(["App"])["Rating"].mean().reset_index())
Data.sort_values(by="Rating",ascending=False,inplace=True)
Data.reset_index(drop=True)
fig = px.bar(Data[:20], y="Rating", x="App", text='App',color="Rating")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=120, uniformtext_mode='hide')
fig.show()
Data_2=pd.DataFrame(df.groupby(["App","Rating"])["Reviews"].mean().reset_index())
Data_2.sort_values(by="Reviews",ascending=False,inplace=True)
Data_2.reset_index(drop=True)
fig = px.bar(Data_2[:20], y="Reviews", x="App", text='Reviews',color="Rating")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=120, uniformtext_mode='hide')
fig.update_layout(title_text="Top 20 Reviewed Apps ")
fig.show()
price=[]

for i in df.Price:
    i=i.split("$")
    if len(i)==1:
        price.append(i[0])
    elif len(i)==2:
        price.append(i[1])

df.Price=price
df.Price=df.Price.astype(float)
Data_3=pd.DataFrame(df.groupby(["App","Rating"])["Price"].mean().reset_index())
Data_3.sort_values(by="Price",ascending=False,inplace=True)
Data_3.reset_index(drop=True)
Data_3=Data_3[Data_3["Price"]<=300.00]
Data_3

fig = px.bar(Data_3[:20], y="Price", x="App", text='Price',color="Rating")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=120, uniformtext_mode='hide')
fig.update_layout(title_text="Top 20 Expensive apps ")
fig.show()
ınstalls_list=[]

for i in df.Installs:
    i=i.split("+")
    ınstalls_list.append(i[0])
df.Installs=ınstalls_list
fixed_ıns=[]
a=""
b=""
c=""

for i in df.Installs:
    i=i.split(",")
    if len(i)==1:
        fixed_ıns.append(i[0])
    elif len(i)==2:
        a= i[0]+i[1]
        fixed_ıns.append(a)
    elif len(i)==3:
        b=i[0]+i[1]+i[2]
        fixed_ıns.append(b)
    elif len(i)==4:
        c=i[0]+i[1]+i[2]+i[3]
        fixed_ıns.append(c)
        
        
        
        
    
df.Installs=fixed_ıns
df.Installs=df.Installs.astype(int)
Data_ıns=pd.DataFrame(df.groupby(["App","Reviews"])["Installs"].mean().reset_index())
Data_ıns.sort_values(by="Reviews",ascending=False,inplace=True)
Data_ıns.reset_index(drop=True)
Data_ıns.drop_duplicates(subset=["App"],inplace=True)
fig = px.bar(Data_ıns[:100], y="Reviews", x="App", text='Installs',color="Installs")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=120, uniformtext_mode='hide')
fig.update_layout(title_text="Top 20 apps that have higher installs and higher reviews than other apps ")
fig.show()
df2016=df[df["Year"]==2016]
df2017=df[df["Year"]==2017]
df2018=df[df["Year"]==2018]

df2018.Type.value_counts().values[0]

year_list=[2016,2017,2018]
paid_count=[]
free_count=[]


for i in year_list:
    x=df[df["Year"]==i]
    paid_count.append(x.Type.value_counts().values[1])
    free_count.append(x.Type.value_counts().values[0])
    
data_t=pd.DataFrame({"Year": year_list,"Free":free_count,"Paid":paid_count})
    
    
    
trace1 = go.Bar(
                x = data_t.Year,
                y = data_t.Free,
                name = "Free apps",
                marker = dict(color = 'rgba(100, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_t.Year)
 
trace2 = go.Bar(
                x = data_t.Year,
                y = data_t.Paid,
                name = "Paid apps",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = data_t.Year)
    
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
fig.update_layout(title_text="Free and Paid apps in 2016,2017 and 2018")
iplot(fig)
dt=pd.DataFrame({"Year":df.Year.value_counts().index,"Count":df.Year.value_counts().values})
dt.sort_values(by="Year",ascending=False,inplace=True)
dt.reset_index(drop=True)
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Year',y='Count',data=dt,color='lime',alpha=0.8)
plt.ylabel("Numbers of apps")
plt.title("How many apps updated in these years ?")
plt.grid()
Df_n=pd.DataFrame({"Content_Rating":df.Content_Rating.value_counts().index,"Count":df.Content_Rating.value_counts().values})
Df_n.head()
labels = Df_n.Content_Rating
values = Df_n.Count

fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0.2, 0.2, 0.2, 0.2])])
fig.show()
labels = df.Size.value_counts().index[:20]
values = df.Size.value_counts().values[:20]

fig = go.Figure(data=[go.Pie(labels=labels, values=values,pull=[0.1 for i in labels ])])
fig.update_layout(title_text="Number of Apps and Their size in All time with Pie chart")

fig.show()
from wordcloud import WordCloud

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='black',
                          width=512,
                          height=384
                         ).generate(" ".join(df.App))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
import plotly.express as px
fig = px.box(df,x="Type" ,y="Rating")
fig.show()
import plotly.express as px
fig = px.box(df,x="Content_Rating" ,y="Rating",color="Content_Rating")
fig.show()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
import plotly.express as px
fig = px.scatter(df, x="Reviews", y="Installs", color="Rating",
                 size='Reviews', hover_data=["App"])
fig.show()