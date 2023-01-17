!pip install dabl 

import dabl
from dabl import plot

from dabl.datasets import load_ames

import matplotlib.pyplot as plt



# load the ames housing dataset

# returns a plain dataframe

data = load_ames()



plot(data, 'SalePrice')

plt.show()
from sklearn.datasets import fetch_openml

from dabl import plot



X, y = fetch_openml('diamonds', as_frame=True, return_X_y=True)



plot(X, y)

plt.show()
from sklearn.datasets import load_wine

from dabl.utils import data_df_from_bunch



wine_bunch = load_wine()

wine_df = data_df_from_bunch(wine_bunch)



plot(wine_df, 'target')

plt.show()
import dabl

import pandas as pd

titanic = pd.read_csv(dabl.datasets.data_path("titanic.csv"))

titanic_df_clean = dabl.clean(titanic, verbose=1)
types = dabl.detect_types(titanic_df_clean)

print(types) 
ec = dabl.SimpleClassifier(random_state=0).fit(titanic, target_col="ticket")
import missingno as msno

import numpy as np
import pandas as pd

database = pd.read_csv("../input/vehicle-collisions/database.csv")

database.info()
%matplotlib inline

collisions = database.replace("nan", np.nan)

msno.matrix(collisions.sample(250))

plt.show()
null_pattern = (np.random.random(1000).reshape((50, 20)) > 0.5).astype(bool)

null_pattern = pd.DataFrame(null_pattern).replace({False: None})

msno.matrix(null_pattern.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ')

plt.show()
msno.bar(collisions.sample(1000))

plt.show()
msno.heatmap(collisions)

plt.show()
msno.dendrogram(collisions)

plt.show()
# installing and importing the library



!pip install pyflux



import pyflux as pf
from pandas_datareader.data import DataReader

from datetime import datetime



a = DataReader('JPM',  'yahoo', datetime(2006,6,1), datetime(2016,6,1))

a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))

a_returns.index = a.index.values[1:a.index.values.shape[0]]

a_returns.columns = ["JPM Returns"]



a_returns.head()
plt.figure(figsize=(15, 5))

plt.ylabel("Returns")

plt.plot(a_returns)

plt.show()
maruti = pd.read_csv("../input/nifty50-stock-market-data/MARUTI.csv")

# Convert string to datetime64

maruti ['Date'] = maruti ['Date'].apply(pd.to_datetime)

maruti.head()
maruti_df = maruti[['Date','VWAP']]



#Set Date column as the index column.

maruti_df.set_index('Date', inplace=True)

maruti_df.head()
plt.figure(figsize=(15, 5))

plt.ylabel("Volume Weighted Average Price'")

plt.plot(maruti_df)

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://www.machinelearningplus.com/wp-content/uploads/2019/02/Equation-3-min.png',width=400,height=400)
my_model = pf.ARIMA(data=maruti_df, ar=4, ma=4, family=pf.Normal())

print(my_model.latent_variables)



result = my_model.fit("MLE")

result.summary()



my_model.plot_z(figsize=(15,5))

my_model.plot_fit(figsize=(15,10))

my_model.plot_predict_is(h=50, figsize=(15,5))

my_model.plot_predict(h=20,past_values=20,figsize=(15,5))
from IPython.display import display

from IPython.display import Image

Image(url = 'https://publichealth.jmir.org/api/download?filename=1aa9764d6789a491635ab5a3238e9da6.png&alt_name=10827-230395-1-PB.png',width=400,height=400)
data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/drivers.csv")

data.index = data['time']

data.loc[(data['time']>=1983.05), 'seat_belt'] = 1

data.loc[(data['time']<1983.05), 'seat_belt'] = 0

data.loc[(data['time']>=1974.00), 'oil_crisis'] = 1

data.loc[(data['time']<1974.00), 'oil_crisis'] = 0
plt.figure(figsize=(15,5))

plt.plot(data.index,data=data)

plt.ylabel('Driver Deaths')

plt.title('Deaths of Car Drivers in Great Britain 1969-84')
model = pf.ARIMAX(data=data, formula='time~1+seat_belt+oil_crisis',

                  ar=1, ma=1, family=pf.Normal())



x = model.fit("MLE")

x.summary()





model.plot_z(figsize=(15,5))

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(15,5))
!pip install bokeh
import bokeh 

bokeh.sampledata.download()
# bokeh packages

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

output_notebook()
N = 4000

x = np.random.random(size=N) * 150

y = np.random.random(size=N) * 150

radii = np.random.random(size=N) * 1.5

colors = [

    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)

]



TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"



p = figure(tools=TOOLS)



p.scatter(x, y, radius=radii,

          fill_color=colors, fill_alpha=0.6,

          line_color=None)



show(p)  # open a browser
from scipy.integrate import odeint

sigma = 10

rho = 28

beta = 8.0/3

theta = 3 * np.pi / 4

def lorenz(xyz, t):

    x, y, z = xyz

    x_dot = sigma * (y - x)

    y_dot = x * rho - x * z - y

    z_dot = x * y - beta* z

    return [x_dot, y_dot, z_dot]

initial = (-10, -7, 35)

t = np.arange(0, 100, 0.006)



solution = odeint(lorenz, initial, t)



x = solution[:, 0]

y = solution[:, 1]

z = solution[:, 2]

xprime = np.cos(theta) * x - np.sin(theta) * y

colors = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B",]

p = figure(title="Lorenz attractor example", background_fill_color="#fafafa")

p.multi_line(np.array_split(xprime, 7), np.array_split(z, 7),

             line_color=colors, line_alpha=0.8, line_width=1.5)



show(p)
def datetime(x):

    return np.array(x, dtype=np.datetime64)



p1 = figure(x_axis_type="datetime", title="Stock Closing Prices")

p1.grid.grid_line_alpha=0.3

p1.xaxis.axis_label = 'Date'

p1.yaxis.axis_label = 'Price'



p1.line(datetime(AAPL['date']), AAPL['adj_close'], color='#A6CEE3', legend_label='AAPL')

p1.line(datetime(GOOG['date']), GOOG['adj_close'], color='#B2DF8A', legend_label='GOOG')

p1.line(datetime(IBM['date']), IBM['adj_close'], color='#33A02C', legend_label='IBM')

p1.line(datetime(MSFT['date']), MSFT['adj_close'], color='#FB9A99', legend_label='MSFT')

p1.legend.location = "top_left"



aapl = np.array(AAPL['adj_close'])

aapl_dates = np.array(AAPL['date'], dtype=np.datetime64)



window_size = 30

window = np.ones(window_size)/float(window_size)

aapl_avg = np.convolve(aapl, window, 'same')



p2 = figure(x_axis_type="datetime", title="AAPL One-Month Average")

p2.grid.grid_line_alpha = 0

p2.xaxis.axis_label = 'Date'

p2.yaxis.axis_label = 'Price'

p2.ygrid.band_fill_color = "olive"

p2.ygrid.band_fill_alpha = 0.1



p2.circle(aapl_dates, aapl, size=4, legend_label='close',

          color='darkgrey', alpha=0.2)



p2.line(aapl_dates, aapl_avg, legend_label='avg', color='navy')

p2.legend.location = "top_left"





show(gridplot([[p1,p2]], plot_width=400, plot_height=400)) # open a browser
from bokeh.sampledata.periodic_table import elements

from bokeh.transform import dodge, factor_cmap





periods = ["I", "II", "III", "IV", "V", "VI", "VII"]

groups = [str(x) for x in range(1, 19)]



df = elements.copy()

df["atomic mass"] = df["atomic mass"].astype(str)

df["group"] = df["group"].astype(str)

df["period"] = [periods[x-1] for x in df.period]

df = df[df.group != "-"]

df = df[df.symbol != "Lr"]

df = df[df.symbol != "Lu"]



cmap = {

    "alkali metal"         : "#a6cee3",

    "alkaline earth metal" : "#1f78b4",

    "metal"                : "#d93b43",

    "halogen"              : "#999d9a",

    "metalloid"            : "#e08d49",

    "noble gas"            : "#eaeaea",

    "nonmetal"             : "#f1d4Af",

    "transition metal"     : "#599d7A",

}



TOOLTIPS = [

    ("Name", "@name"),

    ("Atomic number", "@{atomic number}"),

    ("Atomic mass", "@{atomic mass}"),

    ("Type", "@metal"),

    ("CPK color", "$color[hex, swatch]:CPK"),

    ("Electronic configuration", "@{electronic configuration}"),

]



p = figure(title="Periodic Table (omitting LA and AC Series)", plot_width=1000, plot_height=450,

           x_range=groups, y_range=list(reversed(periods)),

           tools="hover", toolbar_location=None, tooltips=TOOLTIPS)



r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6, legend_field="metal",

           color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))



text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}



x = dodge("group", -0.4, range=p.x_range)



p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)



p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",

       text_font_size="8pt", **text_props)



p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",

       text_font_size="5pt", **text_props)



p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",

       text_font_size="5pt", **text_props)



p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")



p.outline_line_color = None

p.grid.grid_line_color = None

p.axis.axis_line_color = None

p.axis.major_tick_line_color = None

p.axis.major_label_standoff = 0

p.legend.orientation = "horizontal"

p.legend.location ="top_center"

p.hover.renderers = [r] # only hover element boxes



show(p)
!pip install vega_datasets
import altair as alt

from vega_datasets import data



source = data.us_employment()



alt.Chart(source).mark_bar().encode(

    x="month:T",

    y="nonfarm_change:Q",

    color=alt.condition(

        alt.datum.nonfarm_change > 0,

        alt.value("steelblue"),  # The positive color

        alt.value("orange")  # The negative color

    )

).properties(width=600)
source=data.barley()

bars = alt.Chart(source).mark_bar().encode(

    x=alt.X('sum(yield):Q', stack='zero'),

    y=alt.Y('variety:N'),

    color=alt.Color('site')

)

text = alt.Chart(source).mark_text(dx=-15, dy=3, color='white').encode(

    x=alt.X('sum(yield):Q', stack='zero'),

    y=alt.Y('variety:N'),

    detail='site:N',

    text=alt.Text('sum(yield):Q', format='.1f')

)

bars + text
source = data.stocks()

base = alt.Chart(source).properties(width=550)

line = base.mark_line().encode(

    x='date',

    y='price',

    color='symbol'

)

rule = base.mark_rule().encode(

    y='average(price)',

    color='symbol',

    size=alt.value(2)

)

line + rule
x = np.random.normal(size=100)

y = np.random.normal(size=100)



m = np.random.normal(15, 1, size=100)



source = pd.DataFrame({"x": x, "y":y, "m":m})



# interval selection in the scatter plot

pts = alt.selection(type="interval", encodings=["x"])



# left panel: scatter plot

points = alt.Chart().mark_point(filled=True, color="black").encode(

    x='x',

    y='y'

).transform_filter(

    pts

).properties(

    width=300,

    height=300

)



# right panel: histogram

mag = alt.Chart().mark_bar().encode(

    x='mbin:N',

    y="count()",

    color=alt.condition(pts, alt.value("black"), alt.value("lightgray"))

).properties(

    width=300,

    height=300

).add_selection(pts)



# build the chart:

alt.hconcat(

    points,

    mag,

    data=source

).transform_bin(

    "mbin",

    field="m",

    bin=alt.Bin(maxbins=20)

)
source = data.iris()



alt.Chart(source).transform_fold(

    ['petalWidth',

     'petalLength',

     'sepalWidth',

     'sepalLength'],

    as_ = ['Measurement_type', 'value']

).transform_density(

    density='value',

    bandwidth=0.3,

    groupby=['Measurement_type'],

    extent= [0, 8]

).mark_area().encode(

    alt.X('value:Q'),

    alt.Y('density:Q'),

    alt.Row('Measurement_type:N')

).properties(width=300, height=50)


source = data.iris()



alt.Chart(source).transform_fold(

    ['petalWidth',

     'petalLength',

     'sepalWidth',

     'sepalLength'],

    as_ = ['Measurement_type', 'value']

).transform_density(

    density='value',

    bandwidth=0.3,

    groupby=['Measurement_type'],

    extent= [0, 8],

    counts = True,

    steps=200

).mark_area().encode(

    alt.X('value:Q'),

    alt.Y('density:Q', stack='zero'),

    alt.Color('Measurement_type:N')

).properties(width=400, height=100)
source = data.seattle_weather()



line = alt.Chart(source).mark_line(

    color='red',

    size=3

).transform_window(

    rolling_mean='mean(temp_max)',

    frame=[-15, 15]

).encode(

    x='date:T',

    y='rolling_mean:Q'

)



points = alt.Chart(source).mark_point().encode(

    x='date:T',

    y=alt.Y('temp_max:Q',

            axis=alt.Axis(title='Max Temp'))

)



points + line


source = alt.topo_feature(data.world_110m.url, 'countries')



base = alt.Chart(source).mark_geoshape(

    fill='#666666',

    stroke='white'

).properties(

    width=300,

    height=180

)



projections = ['equirectangular', 'mercator', 'orthographic', 'gnomonic']

charts = [base.project(proj).properties(title=proj)

          for proj in projections]



alt.concat(*charts, columns=2)

counties = alt.topo_feature(data.us_10m.url, 'counties')

source = data.unemployment.url



alt.Chart(counties).mark_geoshape().encode(

    color='rate:Q'

).transform_lookup(

    lookup='id',

    from_=alt.LookupData(source, 'id', ['rate'])

).project(

    type='albersUsa'

).properties(

    width=500,

    height=300

)
airports = data.airports.url

states = alt.topo_feature(data.us_10m.url, feature='states')



# US states background

background = alt.Chart(states).mark_geoshape(

    fill='lightgray',

    stroke='white'

).properties(

    width=500,

    height=300

).project('albersUsa')



# airport positions on background

points = alt.Chart(airports).transform_aggregate(

    latitude='mean(latitude)',

    longitude='mean(longitude)',

    count='count()',

    groupby=['state']

).mark_circle().encode(

    longitude='longitude:Q',

    latitude='latitude:Q',

    size=alt.Size('count:Q', title='Number of Airports'),

    color=alt.value('steelblue'),

    tooltip=['state:N','count:Q']

).properties(

    title='Number of airports in US'

)



background + points