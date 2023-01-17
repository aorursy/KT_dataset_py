import pandas as pd

from plotly import graph_objs as go

from ipywidgets import widgets
df=pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')

df=df.drop(["Lat","Long","Province/State"],axis=1)
df.head()
world_temp=df.groupby(df.Date, as_index=False).aggregate(sum)

world_temp["Country/Region"]=["World"]*len(world_temp)

df=pd.concat([df,world_temp],sort=False)
def splitByCountry1(df,country):

    split=df.loc[df["Country/Region"]== country].drop("Country/Region",axis=1)

    split.Date=pd.to_datetime(split.Date)

    split=split.groupby(split.Date).aggregate(sum)

    return split

def splitByCountry2(df,country):

    split=splitByCountry1(df,country)

    L, M, N=[split.Deaths[0]], [split.Confirmed[0]], [split.Recovered[0]]

    for i in range(1,len(split)):

        L.append(split.Deaths[i]-split.Deaths[i-1])

        M.append(split.Confirmed[i]-split.Confirmed[i-1])

        N.append(split.Recovered[i]-split.Recovered[i-1])

    split["New Confirmed"],split["New Recovered"],split["New Deaths"]=M,N,L

    return split.drop(["Confirmed","Recovered","Deaths"],axis=1)
splitByCountry1(df,"Tunisia").tail()
textbox = widgets.Dropdown(

    description="Country:   ",

    value="World",

    options=df["Country/Region"].unique().tolist()

)

since = widgets.IntSlider(

    value=0.0,

    min=0.0,

    max=(pd.to_datetime(max(df.Date))-pd.to_datetime(min(df.Date))).days-10,

    step=1.0,

    description="Since day",

    continuous_update=True

)

keep_C = widgets.Checkbox(

    description="Confirmed",

    value=True

)

keep_R = widgets.Checkbox(

    description="Recovered",

    value=True

)

keep_D = widgets.Checkbox(

    description="Deaths",

    value=True

)

choice = widgets.ToggleButtons(

    options=["Total","Daily"],

    value="Total",

    disabled=False,

    tooltips=["New cases per day", "Total number of cases"],

)

H1 = widgets.HBox(children=[textbox,since])

H2 = widgets.HBox([keep_C,keep_R,keep_D])
df2=splitByCountry1(df,"World")

trace1=go.Scatter(y=df2["Confirmed"],x=df2.index,fill="tozeroy",mode="none",name="Confirmed")

trace2=go.Scatter(y=df2["Recovered"],x=df2.index,fill="tozeroy",mode="none",name="Recovered")

trace3=go.Scatter(y=df2["Deaths"],x=df2.index,fill="tozeroy",mode="none",name="Deaths")

g = go.FigureWidget(data=[trace1,trace2,trace3],layout=go.Layout(title="Covid-19 Statistics - World (Last Update: "+df2.index[-1].strftime("%d %B %Y")+")"))
def response(change):

    x1,x2,x3=[],[],[]

    if (choice.value=="Total"):

        temp_df=splitByCountry1(df,textbox.value)[since.value:]

    else:

        temp_df=splitByCountry2(df,textbox.value)[since.value:]

    if(keep_C.value): x1 = temp_df.iloc[:,0]

    if(keep_R.value): x2 = temp_df.iloc[:,1]

    if(keep_D.value): x3 = temp_df.iloc[:,2]

    with g.batch_update():

        g.data[0].y = x1

        g.data[1].y = x2

        g.data[2].y = x3

        g.layout.title= "Covid-19 Statistics - "+choice.value+"/"+textbox.value+" (Last Update: "+temp_df.index[-1].strftime("%d %B %Y")+")"

        g.layout.xaxis.title = "Date"

        g.layout.yaxis.title = "Number"
textbox.observe(response, names="value")

since.observe(response, names="value")

keep_C.observe(response, names="value")

keep_R.observe(response, names="value")

keep_D.observe(response, names="value")

choice.observe(response, names="value")
V=widgets.VBox([choice])

V.layout.align_items = 'center'

widgets.VBox([g,H1,V,H2])