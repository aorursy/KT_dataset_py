#LOAD PANDAS AND VARIOUS DATA VISUALIZATION LIBRARIES

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
#LOAD A DATE SENSITIVE PANDAS DATAFRAME 

data = pd.read_csv("../input/favorableunfavorable-polling-for-us-candidates/POLITICAL_FAVORABLES_RCP_icod_20200317.csv", parse_dates=True, index_col="DATEFORMATTED")
#PRINT OUT THE DATA TO SEE FIRST COUPLE ROWS

data.head()
#WHAT DOES THE DATA LOOK LIKE

data.info()
# Determine the number of polls per candidate

hue_colors = {"BIDEN":"blue","HRC":"cyan","BERNIE":"green","TRUMP": "red"} 



g = sns.catplot(x="CANDIDATE",data=data,hue = "CANDIDATE",palette=hue_colors,orient="h",kind="count",

                height=8)



g.set(xlabel="Candidate", ylabel="Number of Fav/unfav Polls") 



plt.show()
#another look at this data with a heatmap. Heatmaps in Seaborn are fairly straight forward 

hmapdata =  data.filter(["SOURCE","CANDIDATE"])

hmapdata.head()
#Count the number of polls per candidate by source 

hmapdata = hmapdata.groupby(["SOURCE","CANDIDATE"]).size().reset_index()

hmapdata.columns
hmapdata.SOURCE.unique()
hmapdata = hmapdata.pivot("SOURCE", "CANDIDATE", 0)

hmapdata.head()
hmapdata.loc["PPP (D)"]
fig, ax = plt.subplots(figsize=(10,10))    

ax = sns.heatmap(hmapdata,linewidths=.5,cmap="YlGnBu")
#all candidates and all polls

data = data.sort_index()

g = sns.relplot(data = data.reset_index(), x = 'DATEFORMATTED',y="MARGIN", hue="CANDIDATE",palette=hue_colors,

            kind="scatter",height=8,aspect=1.25)

g.set(xlabel="Poll Date",ylabel="Favorable - Unfavorable Margin",xlim=("2008-12-01","2020-05-17"))

plt.show()
indices = data['CANDIDATE']=="BERNIE"

bernie = data.loc[indices,:] # extract new DataFrame

bernie.head()
# what is the highest favorable for Bernie

bernie.FAVORABLE.max()
# what is the best and worst favroable - unfavorable margin for Bernie

print("Bernie's highest favorable margin was " + str(bernie.MARGIN.max()) + " and Bernie's worst favorable rating was " + str(bernie.MARGIN.min()))
bernie = bernie.sort_index()

g = sns.relplot(data = bernie.reset_index(), x = 'DATEFORMATTED',y="MARGIN",

            kind="scatter",height=8,aspect=1.25,color="green")

g.set(xlabel="Poll Date",ylabel="Favorable - Unfavorable Margin",xlim=("2014-01-01","2020-05-17"))

plt.show()
import plotly

import plotly.graph_objects as go

bernie = bernie.resample('M').mean().ffill()

fig = go.Figure()

fig.add_trace(go.Scatter(

                x=bernie.index,

                y=bernie['FAVORABLE'],

                name="Favorables",

                line_color='black',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=bernie.index,

                y=bernie['UNFAVORABLE'],

                name="Unfavorables",

                line_color='red',

                opacity=0.8))



fig.update_layout(title_text='A Tale of Two Elections: Feeling the Bern No More',

                  xaxis_rangeslider_visible=True,xaxis_title="Year",

    yaxis_title="Avg Monthly Favorables/Unfavorables")



fig.update_layout

fig.show()
#compare Bernie's favorable - unfavorable margin in terms of days out from the general respectively



election_colors = {"2016.0":"purple","2020.0":"orange"} 

h=sns.relplot(data= bernie,x="DAYS_TO_GENERAL_ELECTION", y="MARGIN", hue=bernie["ELECTION_YEAR"].astype(str),kind="line",palette=election_colors,

             height = 8, aspect=1.25)



h.set(xlabel="Days to Election",ylabel="Avg Monthly Favorable - Unfavorable Margin",xlim=(800,100),title= "Comparing Bernie's Favorables - Days out from Election Day")
indices = data['CANDIDATE']=="HRC"

hrc = data.loc[indices,:] # extract new DataFrame

hrc.head()
# what is the highest favorable for Hillary

hrc.FAVORABLE.max()
# what is the best and worst favroable - unfavorable margin for Hillary

print("HRC's highest favorable vs unfavorable margin is " + str(hrc.MARGIN.max()) + " and HRC's worst favorable vs unfavorable margin is" + str(hrc.MARGIN.min()))
hrc= hrc.sort_index()

g = sns.relplot(data = hrc.reset_index(), x = 'DATEFORMATTED',y="MARGIN",

            kind="scatter",height=8,aspect=1.25,color="cyan")

g.set(xlabel="Poll Date",ylabel="Favorable - Unfavorable Margin",xlim=("2009-01-01","2017-01-28"))

plt.show()
# Add shape regions

hrc = hrc.resample('M').mean().ffill()

fig = go.Figure()

fig.add_trace(go.Scatter(

                x=hrc.index,

                y=hrc['FAVORABLE'],

                name="Favorables",

                line_color='black',mode="lines+markers",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=hrc.index,

                y=hrc['UNFAVORABLE'],

                name="Unfavorables",

                line_color='red',mode="lines+markers",

                opacity=0.8))









fig.update_layout(title_text='"The Vast Right-Wing Conspriacy to bring HRC Down"',

                  xaxis_rangeslider_visible=True,xaxis_title="Year",

    yaxis_title="Avg Monthly Favorables/Unfavorables",shapes=[

        # Highlight the months around March 2013

        dict(

            type="rect",

            # x-reference is assigned to the x-values

            xref="x",

            # y-reference is assigned to the plot paper [0,1]

            yref="paper",

            x0="2013-02-01",

            y0=0,

            x1="2013-05-01",

            y1=1,

            fillcolor="LightSalmon",

            opacity=0.3,

            layer="below",

            line_width=0,

            name="HRC Announces Support for Gay Marriage and Gawker expose on email server release",

        )])



fig.update_layout

fig.show()

indices = data['CANDIDATE']=="BIDEN"

biden = data.loc[indices,:] # extract new DataFrame

biden.head()
# what is the highest favorable for Biden

biden.FAVORABLE.max()
# what is the best and worst favroable - unfavorable margin for Biden

print("Biden's highest favorable vs unfavorable margin is " + str(biden.MARGIN.max()) + " and Biden's worst favorable vs unfavorable margin is " + str(biden.MARGIN.min()))
biden= biden.sort_index()

g = sns.relplot(data = biden.reset_index(), x = 'DATEFORMATTED',y="MARGIN",

            kind="scatter",height=8,aspect=1.25,color="blue")

g.set(xlabel="Poll Date",ylabel="Favorable - Unfavorable Margin",xlim=("2018-08-01","2020-04-01"))

plt.show()
# Add shape regions

biden = biden.resample('M').mean().ffill()

fig = go.Figure()

fig.add_trace(go.Scatter(

                x=biden['DAYS_TO_GENERAL_ELECTION'],

                y=biden['MARGIN'],

                name="Biden Favorables - Unfavorables Margin Election 2020",

                line_color='blue',mode="lines",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=hrc['DAYS_TO_GENERAL_ELECTION'],

                y=hrc['MARGIN'],

                name="HRC Favorables - Unfavorables Margin Election 2016",

                line_color='cyan',mode="lines",

                opacity=0.8))





fig.update_layout(title_text="Comparing Biden and HRC's Favorable - Unfavorable Margin in terms of Days to the Election",

                  xaxis_rangeslider_visible=True,xaxis_title="Days to the Election",

    yaxis_title="Monthly Avg Favorable - Unfavorable Margin")



fig.update_xaxes(range=[800, 200])



fig.update_layout

fig.show()

indices = data['CANDIDATE']=="TRUMP"

trump = data.loc[indices,:] # extract new DataFrame

trump.head()
# what is the highest favorable for Trump

trump.FAVORABLE.max()
# what is the best and worst favroable - unfavorable margin for Trump

print("Trump's highest favorable vs unfavorable margin is " + str(trump.MARGIN.max()) + " and Trump's worst favorable vs unfavorable margin is " + str(trump.MARGIN.min()))
trump= trump.sort_index()

g = sns.relplot(data = trump.reset_index(), x = 'DATEFORMATTED',y="MARGIN",

            kind="scatter",height=8,aspect=1.25, color="red")

g.set(xlabel="Poll Date",ylabel="Favorable - Unfavorable Margin",xlim=("2015-05-01","2020-04-01"))

plt.show()
# Add shape regions

trump= trump.resample('M').mean().ffill()

fig = go.Figure()

fig.add_trace(go.Scatter(

                x=trump.index,

                y=trump['FAVORABLE'],

                name="Favorables",

                line_color='black',mode="lines+markers",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=trump.index,

                y=trump['UNFAVORABLE'],

                name="Unfavorables",

                line_color='red',mode="lines+markers",

                opacity=0.8))









fig.update_layout(title_text='Trump Favorables vs Unfavorables',

                  xaxis_rangeslider_visible=True,xaxis_title="Year",

    yaxis_title="Avg Monthly Favorables/Unfavorables")



fig.update_layout

fig.show()

indices = trump['ELECTION_YEAR']==2016

trump2016 = trump.loc[indices,:] # extract new DataFrame
mask = trump['ELECTION_YEAR']==2020

trump2020 = trump.loc[mask,:] # extract new DataFrame
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=biden['DAYS_TO_GENERAL_ELECTION'],

                y=biden['MARGIN'],

                name="Biden Favorables - Unfavorables Margin Election 2020",

                line_color='blue',mode="lines",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=hrc['DAYS_TO_GENERAL_ELECTION'],

                y=hrc['MARGIN'],

                name="HRC Favorables - Unfavorables Margin Election 2016",

                line_color='cyan',mode="lines",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=trump2016['DAYS_TO_GENERAL_ELECTION'],

                y=trump2016['MARGIN'],

                name="Trump Favorables - Unfavorables Margin Election 2016",

                line_color='darkred',mode="lines",

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=trump2020['DAYS_TO_GENERAL_ELECTION'],

                y=trump2020['MARGIN'],

                name="Trump Favorables - Unfavorables Margin Election 2020",

                line_color='red',mode="lines",

                opacity=0.8))





fig.update_layout(title_text="Comparing Biden, HRC and Trump's Favorable - Unfavorable Margin in terms of Days to the Election",

                  xaxis_rangeslider_visible=True,xaxis_title="Days to the Election",

    yaxis_title="Monthly Avg Favorable - Unfavorable Margin")



fig.update_xaxes(range=[800,215])



fig.update_layout

fig.show()
