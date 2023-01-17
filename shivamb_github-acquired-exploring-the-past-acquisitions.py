## load the libraries 
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
df = pd.read_csv("../input/acquisitions.csv")
import squarify
from collections import Counter


## bar plots horizontal
def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
        
    
    trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color, opacity=0.8))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

## bar plots horizontal with x,y
def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color, opacity=0.8))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

## bar plots verticals with x and y 
def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color, opacity=0.8))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
## Generating all visuals for a company 
def generate_all(compdf, compname, color):
    t = compdf[['Company', 'AcquisitionYear']].groupby('AcquisitionYear').agg({'Company' : lambda x : "|".join(x)})
    x = []
    y = []
    z = []
    s = []
    for i, each in enumerate(t.index):
        x1 = each
        y1 = t.values[i][0].split("|")
        for j,comp in enumerate(y1):
            x.append(x1)
            y.append(j+3)
            p = compdf[compdf['Company'] == comp]['Value (USD)'].iloc(0)[0]
            if str(p).lower() == "nan":
                z.append("Company: " + comp)
            else:
                z.append("Company: " + comp +" <br> Amount: $"+ str(int(p)))
            if p > 1000000000:
                s.append(23)
            elif p > 50000000:
                s.append(21)
            elif p > 25000000:
                s.append(19)
            elif p > 12500000:
                s.append(17)
            elif p > 6500000:
                s.append(15)
            elif p > 25000:
                s.append(13)
            else:
                s.append(10)

    trace1 = go.Scatter(x=x, y=y, mode='markers', text=z, marker=dict(size=s, color=color))
    data = [trace1]
    layout = go.Layout(title="All acquisitions By " + compname,  yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            autotick=True,
            ticks='',
            showticklabels=False
        ), height=600)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    

## Generate the Bars for all companies 
def bars():
    ## all
    costly = compdf.sort_values('Value (USD)', ascending = False)
    x = costly['Company'][:15]
    y = costly['Value (USD)'][:15]
    tr1 = bar_ver_noagg(x, y, "Top "+compname+" Acquisitions by Cost",compcol, h=400, lm=50, rt = False)

    t = dict(compdf['AcquisitionYear'].value_counts())
    x = list(t.keys())
    y = list(t.values())
    tr1 = bar_ver_noagg(x[::-1], y[::-1], "Number of Acquisitions by "+compname+" per Year",compcol, h=500, lm=40, rt=True)



    t = dict(compdf['AcquisitionMonth'].value_counts())
    ordd = ['January', 'February', 'March','April','May','June','July','August','September', 'October', 'November','December']
    y1 = [t[i] for i in ordd] 
    tr2 = bar_ver_noagg(ordd, y1, "Number of Acquisitions by "+compname+" Every Month",compcol, h=400, lm=40, rt=True)

    fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = [compname+' Acquisitions - Per Month',compname+' Acquisitions - Per Year' ])
    fig.append_trace(tr2, 1, 1);
    fig.append_trace(tr1, 1, 2);
    fig['layout'].update(height=400, showlegend=False);
    iplot(fig); 


    col = "AcquisitionMonth"
    data = []
    for each in compdf[col].value_counts().index:
        y1 = compdf[compdf[col]==each]['Value (USD)'].dropna()
        trace0 = go.Box(y=y1, name = each, marker = dict(color = compcol))
        data.append(trace0)

    layout = dict(title="Acquisition Cost per Month - "+compname, height=400, showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

# treemap 1
def createhmap1(compcol):
    t = compdf['Country'].value_counts()
    keys = t.index[1:] 
    vals = t.values[1:]

    x = 0.
    y = 0.
    width = 100.
    height = 100.
    colcnt = 0
    values = vals

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    color_brewer = [compcol]
    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 5, color="#fff" ),
                fillcolor = color_brewer[colcnt]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = list(keys)[counter] +" ("+ str(values[counter]) + ")",
                showarrow = False
            )
        )
        counter = counter + 1
        colcnt+=1
        if colcnt >= len(color_brewer):
            colcnt = 0

    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ], 
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(v)+" ("+str(values[k])+" )" for k,v in enumerate(keys) ], 
        mode = 'text',
    )

    layout = dict(
        height=450, 
        xaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                autotick=True,
                ticks='',
                showticklabels=False
            ),
        yaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                autotick=True,
                ticks='',
                showticklabels=False
            ),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        title="Countries (Apart from USA) from where " + compname + " acquired"
    )

    figure = dict(data=[trace0], layout=layout)
    iplot(figure, filename='squarify-treemap')

## treemap 2 
def createhmap2(compcol):
    txt = " ".join(compdf['Business']).lower()
    wrds = txt.split()
    words = []
    for each in wrds:
        if each.endswith("s"):
            words.append(each[:-1])
        else:
            words.append(each)
    t = Counter(words)
    m = [x for x in t.most_common(30) if x[0] not in STOPWORDS]

    keys = [m1[0] for m1 in m]
    vals = [m1[1] for m1 in m] 

    x = 0.
    y = 0.
    width = 100.
    height = 100.
    colcnt = 0
    values = vals

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    color_brewer = [compcol]
    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 5, color="#fff" ),
                fillcolor = color_brewer[colcnt]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = list(keys)[counter] +" ("+ str(values[counter]) + ")",
                showarrow = False
            )
        )
        counter = counter + 1
        colcnt+=1
        if colcnt >= len(color_brewer):
            colcnt = 0

    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ], 
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(v)+" ("+str(values[k])+" )" for k,v in enumerate(keys) ], 
        mode = 'text',
    )

    layout = dict(
        height=450, 
        xaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                autotick=True,
                ticks='',
                showticklabels=False
            ),
        yaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                autotick=True,
                ticks='',
                showticklabels=False
            ),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        title="Business Use-Cases of Acquired Companies by " + compname
    )

    figure = dict(data=[trace0], layout=layout)
    iplot(figure, filename='squarify-treemap')
compcol = "#bd4ff9"
compname = "Microsoft"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "#fc2054"
compname = "Google"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "#66ff87"
compname = "IBM"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "#7b86ed"
compname = "Yahoo"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "#f26a93"
compname = "Apple"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "orange"
compname = "Twitter"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
compcol = "#40fcdc"
compname = "Facebook"
compdf = df[df['ParentCompany'] == compname] 
generate_all(compdf, compname, compcol)
bars()
createhmap1(compcol)
## Business Keywords - Microsoft
compcol = "#bd4ff9"
compname = "Microsoft"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - Google
compcol = "#fc2054"
compname = "Google"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - IBM
compcol = "#66ff87"
compname = "IBM"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - Yahoo
compcol = "#7b86ed"
compname = "Yahoo"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - Twitter
compcol = "orange"
compname = "Twitter"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - Apple
compcol = "#f26a93"
compname = "Apple"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)

## Business Keywords - Facebook
compcol = "#40fcdc"
compname = "Facebook"
compdf = df[df['ParentCompany'] == compname] 
createhmap2(compcol)
