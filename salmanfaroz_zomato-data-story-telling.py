import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
init_notebook_mode(connected=True)
import plotly
plotly.offline.init_notebook_mode (connected = True)

warnings.filterwarnings("ignore")

%matplotlib inline
ds=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
ds["rate"]=ds["rate"].fillna(0)
li=[]
for i in range(0,len(ds)):
    a=ds.iloc[i,5]
    try:
        b=float(a[:-2])
        li.append(b)
    except:
        li.append(0)
ds["rate"]=li
res=ds.nlargest(30,["votes","rate"])
grp=res.groupby("name")
res=grp.mean()
fig = px.bar(res, y=res.rate, x=res.index, text=res.votes)
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
from textblob import TextBlob
sent=[]
for i in range(0,len(ds)):
    analysis = TextBlob(ds.loc[i,"reviews_list"])
    if analysis.sentiment.polarity > 0:
        sent.append("positive")
    elif analysis.sentiment.polarity == 0:
        sent.append("Neutral")
    else:
        sent.append("negative")
ds["reviews_list"]=sent
#ds=pd.read_csv('processed.csv')
ds["Count"]=1
pos=ds[ds["reviews_list"]=="positive"]
neg=ds[ds["reviews_list"]=="negative"]
cnt_pos=pos.groupby("listed_in(type)").sum()
cnt_neg=neg.groupby("listed_in(type)").sum()
fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = 0,
    delta = {'reference': 0},
    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]},
    title = {'text': "Neutral"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 300]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 170},
        'steps': [
            {'range': [0, 150], 'color': "gray"},
            {'range': [150, 250], 'color': "lightgray"}],
        'bar': {'color': "black"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = 4324,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Negative"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 5000]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 50},
        'steps': [
            {'range': [0, 200], 'color': "gray"},
            {'range': [200, 500], 'color': "lightgray"}],
        'bar': {'color': "black"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge+delta", value = 39629,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text' :"Positive"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 51717]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 210},
        'steps': [
            {'range': [0, 25000], 'color': "gray"},
            {'range': [25000, 51717], 'color': "lightgray"}],
        'bar': {'color': "black"}}))
fig.update_layout(height = 400 , margin = {'t':0, 'b':0, 'l':0})

fig.show()
labels = list(cnt_pos.index)

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=list(cnt_pos.Count), name="Positive"),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=list(cnt_neg.Count), name="Negative"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    annotations=[dict(text='Positive', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='Negative', x=0.82, y=0.5, font_size=20, showarrow=False)])
fig.show()

to=ds.groupby(["listed_in(type)","name"]).sum()

lis=list(to.index)

val1=[]
for i in range(0,len(lis)):
    val1.append(lis[i][0])


st=set(val1)

type_rest=[]
cnt=[]
for i in st:
    type_rest.append(i)
    cnt.append(val1.count(i))
    
    

type_rest,cnt
dff=pd.DataFrame()
dff["Types of Zomato Restaurants in Bangalore"]=type_rest
dff["Count of Restaurants in Bangalore"]=cnt
fig = px.bar(dff, x="Types of Zomato Restaurants in Bangalore", y="Count of Restaurants in Bangalore",
         height=400)
fig.show()
cost=[]
for i in range (0,len(ds)):
    it=ds.loc[i,"approx_cost(for two people)"]
    try:
        a=it.replace(",","")
        a=float(a)
        cost.append(a)
    except:
        cost.append(it)
ds["approx_cost(for two people)"]=cost
top_cost=ds.nsmallest(15,"approx_cost(for two people)")
top_cst=top_cost.loc[:,["dish_liked","approx_cost(for two people)","listed_in(type)","listed_in(city)","name"]]

fig = px.funnel(top_cst, x='approx_cost(for two people)', y='name', color='listed_in(type)')
fig.show()
cost=[]
for i in range (0,len(ds)):
    it=ds.loc[i,"approx_cost(for two people)"]
    try:
        a=it.replace(",","")
        a=float(a)
        cost.append(a)
    except:
        cost.append(it)
ds["approx_cost(for two people)"]=cost
top_cost=ds.nlargest(10,"approx_cost(for two people)")
top_cst=top_cost.loc[:,["dish_liked","approx_cost(for two people)","listed_in(type)","listed_in(city)","name"]]

fig = px.funnel(top_cst, x='approx_cost(for two people)', y='name', color='listed_in(type)')
fig.show()
fin=ds.loc[:,["rate","listed_in(type)","approx_cost(for two people)"]]
fig = px.histogram(fin, x="rate", color="listed_in(type)")
fig.show()
df = px.data.gapminder()
df_2007 = df.query("year==2007")
fig = px.scatter(ds,
                     x="rate", y="approx_cost(for two people)", size="votes", color="reviews_list",
                     log_x=True, size_max=60,
                     template="plotly_dark")
fig.show()

order=ds.groupby(["online_order","book_table"]).sum()
del order["Unnamed: 0"]
#del order["Count"]
del order["approx_cost(for two people)"]
order
fig =go.Figure(go.Sunburst(
    labels=["type", "Online_order","Book_table","Yes","Yes","No","No"],
    parents=["", "type","type","Online_order","Book_table" , "Online_order","Book_table" ],
    values=[51717, 29283, 22434, 26639, 3805, 2644, 18629],
))
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
#51717, 29283, 22434
fig.show()
