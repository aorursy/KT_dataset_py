# section 1 importing Libs:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
# my style:
sns.set(style= "whitegrid")

# My favourite Library for visualisation 
from plotly import __version__
import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

cf.go_offline()

import plotly.figure_factory as ff
import plotly.offline as py
##for online plotting use import plotly.plotly as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
df = pd.read_csv("../input/pricepersqft.csv")
df_rent = pd.read_csv("../input/price.csv")
df.head(10)
df_rent.head(10)
df.describe()
df_rent.describe()
print("\n"+"In the month of November 2010, Maximum and Minimum Price Per Square Feet ")
print(df[df["November 2010"]==df["November 2010"].max()][["Metro","County", "November 2010"]])
print(df[df["November 2010"]==df["November 2010"].min()][["City","Metro", "County", "November 2010"]])
print("____________________________________________________________")

print("\n"+"In the month January 2011, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2011"]==df["January 2011"].max()][["City", "County", "January 2011"]])
print(df[df["January 2011"]==df["January 2011"].min()][["City","Metro", "County", "January 2011"]])
print("____________________________________________________________")

print("\n"+"In the month January 2012, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2012"]==df["January 2012"].max()][["City", "County", "January 2012"]])
print(df[df["January 2012"]==df["January 2012"].min()][["City","Metro", "County", "January 2012"]])
print("____________________________________________________________")

print("\n"+"In the month January 2013, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2013"]==df["January 2013"].max()][["City", "County", "January 2013"]])
print(df[df["January 2013"]==df["January 2013"].min()][["City","Metro", "County", "January 2013"]])
print("____________________________________________________________")

print("\n"+"In the month January 2014, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2014"]==df["January 2014"].max()][["City", "County", "January 2014"]])
print(df[df["January 2014"]==df["January 2014"].min()][["City","Metro", "County", "January 2014"]])
print("____________________________________________________________")

print("\n"+"In the month January 2015, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2015"]==df["January 2015"].max()][["City", "County", "January 2015"]])
print(df[df["January 2015"]==df["January 2015"].min()][["City","Metro", "County", "January 2015"]])
print("____________________________________________________________")

print("\n"+"In the month January 2016, Maximum and Minimum Price Per Square Feet ")
print(df[df["January 2016"]==df["January 2016"].max()][["City", "County", "January 2016"]])
print(df[df["January 2016"]==df["January 2016"].min()][["City","Metro", "County", "January 2016"]])
print("____________________________________________________________")



state_count = df["State"].value_counts()

trace = go.Bar(
    x=state_count.index,
    y=state_count.values,
    marker=dict(
        color = (["lightsteelblue", "lightyellow", "lime", "limegreen",
            "linen", "magenta", "maroon", "mediumaquamarine",
            "mediumblue", "mediumorchid", "mediumpurple","mediumblue", "mediumorchid", "mediumpurple", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender","lightyellow", "lime", "limegreen",
            "linen", "magenta", "maroon", "mediumaquamarine",
            "mediumblue", "mediumorchid", "mediumpurple", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender"])))
layout = go.Layout(
    title='States with Highest listing', yaxis = dict(title = 'Frequency'))
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)




label = state_count.index
size = state_count.values
colors = ["lightsteelblue", "lightyellow", "lime", "limegreen",
            "linen", "magenta", "maroon"]
trace = go.Pie(labels=label, 
               values=size, 
               marker=dict(colors=colors))
layout = go.Layout(
    title='State Distribution')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)



#plt.figure(figsize = (15, 8))
#sns.set_context("paper",font_scale= 2)
#sns.barplot(state_count.index, state_count.values, order = state_count.index)
#plt.xlabel("States")
#plt.xticks(rotation = 90)
#plt.ylabel("Total Number of Listings")
#plt.tight_layout()
#plt.show()
# Highest 20 Metro's
metro_count = df["Metro"].value_counts().head(20)

trace = go.Bar(
    x=metro_count.index,
    y=metro_count.values,
    marker=dict(
        color = (["lightblue", "lightyellow", "lime", "limegreen",
            "linen", "magenta", "maroon", "mediumaquamarine",
            "mediumblue", "mediumorchid", "mediumpurple","mediumblue", "mediumorchid", "mediumpurple", "hotpink", "indianred", "indigo",
            "ivory", "khaki", "lavender" ])))

layout = go.Layout(
    title='20 Metro with Highest listing', yaxis = dict(title = 'Frequency'))

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

label = metro_count.index
size = metro_count.values
colors = ['skyblue', 'orange', '#96D38C', '#D0F9B1']
trace = go.Pie(labels=label, 
               values=size, 
               marker=dict(colors=colors))
layout = go.Layout(
    title='Top 20 Metro Distribution')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# This block of code for Metro listing count plot is optional. (for my github page) 
# As plotly does not work on github.
#plt.figure(figsize=(25,10))
#sns.set_context("paper",font_scale= 2)
#sns.barplot(x=metro_count.index, y= metro_count.values, order= metro_count.index )
#plt.xlabel("Metro")
#plt.xticks(rotation = 90)
#plt.ylabel("Total Number of Listings")
#plt.tight_layout()
#plt.show()
years = list(set([y.split( )[1] for y in df.columns[6:]]))
months = df.columns[6:]
# Getting a dataframe of boston city only:
boston = df[df["Metro"]=="Boston"]
boston.head()
boston_r = df_rent[df_rent["Metro"]=="Boston"]
boston_r.head()
boston_r = df_rent[df_rent["Metro"]=="Boston"]
# Let's see most expensive & least expensice city in boston:
print(" Highest Price Per Square Feet for Jan'10 and Jan'17")
print(boston[boston["November 2010"] == boston["November 2010"].max()][["City", "Metro", "November 2010"]])
print(boston[boston["January 2011"] == boston["January 2011"].max()][["City", "Metro", "January 2011"]])
print(boston[boston["January 2017"] == boston["January 2017"].max()][["City", "Metro", "January 2017"]])
print("_______________________________________________________")

print(" Lowest Price Per Square Feet for Jan'10 and Jan'17")
print(boston[boston["November 2010"] == boston["November 2010"].min()][["City", "Metro", "November 2010"]])
print(boston[boston["January 2011"] == boston["January 2011"].min()][["City", "Metro", "January 2011"]])
print(boston[boston["January 2017"] == boston["January 2017"].min()][["City", "Metro", "January 2017"]])
print("________________________________________________________")
#Variable assig
bos_pi = boston["County"].value_counts()
colors = ["lightblue", "lightyellow", "lime", "limegreen","magenta", "maroon", "mediumaquamarine"]

# Pie chart
trace = go.Pie(labels=bos_pi.index,
              values = bos_pi.values, 
              marker= dict(colors = colors))

layout = go.Layout(title="Distribution in Boston")

data =[trace]
fig = go.Figure(data=data, layout = layout)
py.iplot(fig)

#For Scatter plot
trace = go.Scatter(x = months, 
                  y= np.nanmedian(boston[months], axis = 0),
                    mode='markers', marker=dict(size=3,color = ('orange')),
                   name = "Boston Median PPSFT")



#For Scatter plot
trace1 = go.Scatter(x = months, 
                  y= np.nanmedian(boston_r[months], axis = 0),
                    mode='markers', marker=dict(size=4,color = ('red')), 
                   name = "Boston Median Rent")


fig = tools.make_subplots(rows=1, cols=2,
                          subplot_titles=('PPSFT Boston','Median Rent Boston'))

fig.append_trace(trace, 1, 1)
fig.append_trace(trace1, 1, 2)

layout = go.Layout(title="Median index price of boston", 
                   xaxis= dict(title= "Months"),yaxis=dict(title="PPSFT"))

fig['layout'].update(showlegend=False, title='Price in Boston Per Square Feet VS Rent')
py.iplot(fig)


ny = df[df["Metro"]=="New York"]
ny.head(1)
ny_nj = ny.groupby("State")[months].median()


ny_rent = df_rent[df_rent["Metro"]=="New York"]
ny_rent.head(1)
ny_nj_rent = ny.groupby("State")[months].median()


stat1 = list (set([x for x in ny["State"]]))
#np.median(ny[ny["State"]==stat[0]][months], axis = 0)

# For Scatter Plot 
trace1 = go.Scatter(x = months, 
                  y= np.nanmedian(ny[ny["State"]==stat1[2]][months], axis = 0),
                    mode='markers', marker=dict(size=5,color = ('aqua')), 
                   name = "New York")
trace2 = go.Scatter(x = months, 
                  y= np.nanmedian(ny[ny["State"]==stat1[0]][months], axis = 0),
                    mode='markers', marker=dict(size=5,color = ('navy')),
                   name = "New Jersey")

layout = go.Layout(title="Median PPSFT price of NY, NJ", 
                   xaxis= dict(title= "Months"),yaxis=dict(title="PPSFT"))

data = [trace1, trace2]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)





trace1 = go.Scatter(x = months, 
                  y= np.nanmedian(ny_rent[ny_rent["State"]==stat1[2]][months], axis = 0),
                    mode='markers', marker=dict(size=5,color = ('aqua')), 
                   name = "New York")
trace2 = go.Scatter(x = months, 
                  y= np.nanmedian(ny_rent[ny_rent["State"]==stat1[0]][months], axis = 0),
                    mode='markers', marker=dict(size=5,color = ('navy')),
                   name = "New Jersey")



fig = tools.make_subplots(rows= 1 , cols=2, subplot_titles=('Median NY Rent','Median NJ Rent'))
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1,2)

fig['layout'].update(showlegend=False, title='Median Rent Price of New York VS New Jersey')
py.iplot(fig)




# Box Plot block of code:
ny_gr = ny.groupby("State")[months].median()
print(ny_gr)
trace0 = go.Box(y=ny_gr.loc["NJ"],name="New Jersey",fillcolor='navy')
trace1 = go.Box(y=ny_gr.loc["NY"],name="New York",fillcolor='lime')
trace2 = go.Box(y=ny_gr.loc["PA"],name="Pensylvania",fillcolor='aqua')

layout = go.Layout(title  =" Boxplot of NY, NJ, PA")
data = [trace0, trace1, trace2]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
states = list(df["State"].unique())
state_group = df.groupby("State")[months].median().reset_index()
state_group.head(5)
trace0 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="NY"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "NY")

trace1 =go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="CA"][months], axis = 0), mode='markers', marker=dict(size=3),
                   name = "CA")
trace2 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="HI"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "HI")

trace3 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="DC"][months], axis = 0), 
                    mode='markers', marker=dict(size=3),
                    name = "DC")

trace4 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="AZ"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "AZ")

trace5 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="FL"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "FL")

trace6 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="TX"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "TX")

trace7 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="IL"][months], axis = 0), 
                    mode='markers', marker=dict(size=3),
                    name = "IL")

trace8 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="NC"][months], axis = 0), 
                    mode='markers', marker=dict(size=3),
                    name = "NC")

trace9 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="NV"][months], axis = 0),
                    mode='markers', marker=dict(size=3),
                    name = "NV")

trace10 = go.Scatter(x= months, 
                    y = np.nanmedian(df[df["State"]=="OK"][months], axis = 0),
                     mode='markers', marker=dict(size=3),
                     name = "OK")

layout = go.Layout(title = "Median PPSFT for top 20 States", xaxis= dict(title = "PPSFT"),
                  yaxis = dict(title = "Months"))
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
fig =go.Figure(data=data, layout = layout) 
py.iplot(fig)



#Matplotlib plot
plt.figure(figsize=(17,22))

for st in states:
    st_pick = df[df["State"] == st][months]
    plt.plot(months, np.nanmedian(st_pick, axis=0), label = st)

plt.title("Median PPSFT for all States")    
plt.xlabel("Months")
plt.ylabel("PPSFT")
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.1, 1), loc = 2, borderaxespad = 0)
plt.show()
plt.figure(figsize = (17, 22))
for s in states:
    pr = df[df["State"] == s]
    r = min(pr["Population Rank"])
    pr = pr[pr["Population Rank"] == r]
    label = "{} {}".format(s, pr["City"].unique())
    pr = pr[months]
    plt.plot(pr.columns, np.transpose(pr.values), label = label)
    
plt.title("PPSFT in Largest Population Cities by State", fontsize = 25)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()


trace0 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="NY"][months], axis = 0), name = "NY")

trace1 =go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="CA"][months], axis = 0), name = "CA")
trace2 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="HI"][months], axis = 0), name = "HI")

trace3 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="DC"][months], axis = 0), name = "DC")

trace4 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="AZ"][months], axis = 0), name = "AZ")

trace5 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="FL"][months], axis = 0), name = "FL")

trace6 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="TX"][months], axis = 0), name = "TX")

trace7 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="IL"][months], axis = 0), name = "IL")

trace8 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="NC"][months], axis = 0), name = "NC")

trace9 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="NV"][months], axis = 0), name = "NV")

trace10 = go.Scatter(x= months, 
                    y = np.nanmedian(df_rent[df_rent["State"]=="OK"][months], axis = 0), name = "OK")

layout = go.Layout(title = "Median RENT for top 20 States", xaxis= dict(title = "RENT"),
                  yaxis = dict(title = "Months"))
data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
fig =go.Figure(data=data, layout = layout) 
py.iplot(fig)



#Matplotlib plot
plt.figure(figsize=(17,22))

for st in states:
    st_pick = df_rent[df_rent["State"] == st][months]
    plt.plot(months, np.nanmedian(st_pick, axis=0), label = st)

plt.title("Median RENT for all States", fontsize = 20)    
plt.xlabel("Months", fontsize = 20)
plt.ylabel("RENT", fontsize = 20)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.1, 1), loc = 2, borderaxespad = 0)
plt.show()
cal= df[df["State"]=="CA"]
cal.head(10)
cal_met= cal["Metro"].unique()
#cal_met

plt.figure(figsize=(17,22))
for met in cal_met:
    met_price = cal[cal["Metro"]== met][months]
    plt.plot(months, np.nanmedian(met_price, axis = 0), label = met)


plt.title("Median PPSFT of Metros in California ", fontsize =20)    

plt.xlabel("Months", fontsize = 20)
plt.ylabel("PPSFT", fontsize = 20)
plt.xticks(rotation = 90)
plt.legend(bbox_to_anchor = (1.1, 1), loc = 2, borderaxespad = 0)
plt.show()

