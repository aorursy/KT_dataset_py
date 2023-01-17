import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # collection of functions for scientific and publication-ready visualization
import numpy as np # linear algebra
import json
import bokeh
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.widgets import Select
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import output_notebook
output_notebook() # output_file("project.html")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
gdp_data = pd.read_csv('../input/gdp2017/GDP.csv',skiprows = 3) # matrixds_project_files
gdp_data.drop(['Unnamed: 62'],axis=1, inplace=True) # Drop the last column Unnamed
# gdp_data.info()
gdp_data.shape # 264 countries in 2018
gdp_data.head(3)
id_vars=['Country Name','Country Code', 'Indicator Name', 'Indicator Code']
df = pd.melt(frame=gdp_data, id_vars=id_vars, var_name='year', value_name='GDP') # country_x_2018_forecast
# df.describe()
df['year'] = df['year'].astype(float) # convert from object to float
df.info() # confirm data types
df.shape
df = df.dropna() # drop rows where GDP is NaN
df.shape
df.head(50)
df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
df.rename(columns={'Country Name':'Country'}, inplace=True)
df.head(3)
df['Country'].value_counts() # Pre-demographic dividend?
values = ['Arab World',
          'Caribbean small states',
          'Central Europe and the Baltics',
          'Early-demographic dividend',
          'East Asia & Pacific (excluding high income)',
          'Early-demographic dividend',
          'East Asia & Pacific',
          'East Asia & Pacific (IDA & IBRD countries)',
          'Europe & Central Asia',
          'Europe & Central Asia (IDA & IBRD countries)',
          'Europe & Central Asia (excluding high income)',
          'Euro area',
          'European Union',
          'Fragile and conflict affected situations',
          'Heavily indebted poor countries (HIPC)',
          'High income',
          'IBRD only',
          'IDA & IBRD total',
          'IDA total',
          'IDA blend',
          'IDA only',
          'Late-demographic dividend',
          'Latin America and Caribbean',
          'Latin America & Caribbean',
          'Latin America & Caribbean (excluding high income)',
          'Latin America & the Caribbean (IDA & IBRD countries)',
          'Lower middle income',
          'Low & middle income',
          'Middle income',
          'Middle East & North Africa (IDA & IBRD countries)',
          'Middle East & North Africa',
          'Middle East & North Africa (excluding high income)',
          'North America',
          'OECD members',
          'Pacific island small states',
          'Post-demographic dividend',
          'Pre-demographic dividend',
          'South Asia (IDA & IBRD)',
          'Sub-Saharan Africa (IDA & IBRD countries)',
          'Sub-Saharan Africa (excluding high income)',
          'Sub-Saharan Africa',
          'Small states',
          'Upper middle income',
          'World']
for i in range(0, 60):
    for value in values:
        condition = df[df.Country == value].index
        df.drop(condition, inplace=True)

# df[df['column name'].map(lambda x: str(x)!=".")]

# df.where(m, -df)
df.head(50)
df.tail(50)
df.shape
filename = 'GDP_tidy.csv'
df.to_csv(filename, index=False)
print("{} saved".format(filename))
# Adding a default
country = 'Mexico'
filter = df['Country'] != country
dfmx = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfmx.tail(5)
# Adding a default
country = 'United States'
filter = df['Country'] != country
dfus = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfus.tail(5)
# Adding a default
country = 'Spain'
filter = df['Country'] != country
dfsp = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfsp.tail(5)
# Adding a default
country = 'Canada'
filter = df['Country'] != country
dfca = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfca.tail(5)
# Adding a default
country = 'China'
filter = df['Country'] != country
dfch = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfch.tail(5)
# Adding a default
country = 'India'
filter = df['Country'] != country
dfin = df.drop(df[filter].index, inplace=False) # filter by country
# df.shape
dfin.tail(5)
from scipy import stats
X = dfmx.year
y = dfmx.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP Mexico (2018) = $1,131,888,421,568.4062 MXN
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP Mexico (2018) = ${} MXN'.format(y_pred))
print('SciPy')
plt.scatter(X, y, color='green') # Scatter Plot
plt.plot(X, model, color='red') # linestyle='dashed', marker='o', markersize=12
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP Mexico data (1960-2017)'])
plt.show()
from scipy import stats
X = dfus.year
y = dfus.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP US (2018) = $16,904,994,673,321.25 USD
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP US (2018) = ${} USD'.format(y_pred))
print('SciPy')
plt.scatter(X, y) # Scatter Plot
plt.plot(X, model, color='red') # linestyle='dashed', marker='o', markersize=12, markerfacecolor='blue'
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP US data (1960-2017)'])
plt.show()
from scipy import stats
X = dfsp.year
y = dfsp.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP SPAIN (2018) = $1,378,228,907,914.7578
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP SPAIN (2018) = ${}'.format(y_pred))
print('SciPy')
plt.scatter(X, y, color='orange') # Scatter Plot
plt.plot(X, model, color='red') # linestyle='dashed', marker='o', markersize=12
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP SPAIN data (1960-2017)'])
plt.show()
from scipy import stats
X = dfca.year
y = dfca.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP CANADA (2018) = $1,568,074,796,765.2188
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP CANADA (2018) = ${}'.format(y_pred))
print('SciPy')
plt.scatter(X, y, color='red') # Scatter Plot
plt.plot(X, model, color='black') # linestyle='dashed', marker='o', markersize=12
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP CANADA data (1960-2017)'])
plt.show()
from scipy import stats
X = dfin.year
y = dfin.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP INDIA (2018) = $1,523,808,381,383.0781
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP INDIA (2018) = ${}'.format(y_pred))
print('SciPy')
plt.scatter(X, y, color='orange') # Scatter Plot
plt.plot(X, model, color='green') # linestyle='dashed', marker='o', markersize=12
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP INDIA data (1960-2017)'])
plt.show()
from scipy import stats
X = dfch.year
y = dfch.GDP
slope, intercept, r, p, std_err = stats.linregress(X, y) # scipy
def modelPrediction(x):
  return slope * x + intercept
# Model Prediction GDP CHINA (2018) = $6,347,500,525,036.9375
model = list(map(modelPrediction, X)) # scipy
x_pred = 2018
y_pred = modelPrediction(x_pred)
print('Model Prediction GDP CHINA (2018) = ${}'.format(y_pred))
print('SciPy')
plt.scatter(X, y, color='red') # Scatter Plot
plt.plot(X, model, color='orange') # linestyle='dashed', marker='o', markersize=12
plt.ylim(ymin=0) # starts at zero
plt.legend(['Model Prediction using Linear Regression', 'GDP CHINA data (1960-2017)'])
plt.show()
import sklearn
from sklearn.linear_model import LinearRegression
print('MEXICO')
x = dfmx[['year']].values
y = dfmx.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
print('UNITED STATES')
x = dfus[['year']].values
y = dfus.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
print('SPAIN')
x = dfsp[['year']].values
y = dfsp.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
print('CANADA')
x = dfca[['year']].values
y = dfca.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
print('INDIA')
x = dfin[['year']].values
y = dfin.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
print('CHINA')
x = dfch[['year']].values
y = dfch.GDP.values
regr = sklearn.linear_model.LinearRegression()
model = regr.fit(x,y) # SciKit-Learn
score = regr.score(x, y)
print('score = {}'.format(score))
coef = regr.coef_
print('coef = {}'.format(coef)) # 1.0
intercept = regr.intercept_
print('intercept = {}'.format(intercept)) # 3.0000...
y_pred = model.predict(x)
print('SciKit-Learn')
plt.scatter(x, y, color='gray') # sklearn
plt.plot(x, y_pred, color='orange') # model
plt.ylim(0) # start at zero
plt.show()
# MEXICO
x = dfmx.year
y = dfmx.GDP
# create a new plot with a title and axis labels
p = figure(title="MEX:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpmx = y[15200]
gdpmx = gdpmx / 1000000000000
round(gdpmx, 2)
gdpmx
# US
x = dfus.year
y = dfus.GDP
# create a new plot with a title and axis labels
p = figure(title="US:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpus = y[15297]
gdpus = gdpus / 1000000000000
round(gdpus, 2)
gdpus
# SP
x = dfsp.year
y = dfsp.GDP
# create a new plot with a title and axis labels
p = figure(title="SPAIN:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpsp = y[15116]
gdpsp = gdpsp / 1000000000000
round(gdpsp, 2)
gdpsp
# CA
x = dfca.year
y = dfca.GDP
# create a new plot with a title and axis labels
p = figure(title="Canada:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpca = y[15081]
gdpca = gdpca / 1000000000000
round(gdpca, 2)
gdpca
# IN
x = dfin.year
y = dfin.GDP
# create a new plot with a title and axis labels
p = figure(title="India:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpin = y[15155]
gdpin = gdpin / 1000000000000
round(gdpin, 2)
gdpin
# CH
x = dfch.year
y = dfch.GDP
# create a new plot with a title and axis labels
p = figure(title="China:GDP-by-year", x_axis_label='x', y_axis_label='y')
# add a line renderer with legend and line thickness
p.line(x, y, line_width=2) # , legend_label="Temp."
# show the results
show(p) # output_file("lines.html") # output to static HTML file
gdpch = y[15086]
gdpch = gdpch / 1000000000000
round(gdpch, 2)
gdpch
# Fixing random state for reproducibility
plt.rcdefaults()
fig, ax = plt.subplots()
y = ('United States', 'China', 'India', 'Canada', 'Spain', 'Mexico')
y_pos = np.arange(len(y))
x = (gdpus, gdpch, gdpin, gdpca, gdpsp, gdpmx)
ax.barh(y_pos, x, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(y)
ax.invert_yaxis() # labels read top-to-bottom
ax.set_xlabel('GDP')
ax.set_title('GDP per Country 2017')
for i, v in enumerate(x):
    ax.text(v + 1, i, str(v), color='black', va='center', fontweight='normal')
plt.show()
with open('../input/global/countries.json','r') as f:
    geodata = json.load(f)
f.close()
geodata_features = geodata['features']
country_xs = []
country_ys = []
country_names = []
country_num_users = []
country_colors = [] # ?

#loop through each of the countries
for aCountry in geodata_features:
    cName = aCountry['properties']['name']
    geometry_type = aCountry['geometry']['type']
    
    #countries that have land masses seperated by water have multiple polygons
    if geometry_type == "MultiPolygon":
        for poly_coords in aCountry['geometry']['coordinates']:
            coords = poly_coords[0]
            country_names.append(cName)
            country_xs.append(list(map(lambda x:x[0],coords)))
            country_ys.append(list(map(lambda x:x[1],coords)))
            country_colors.append("purple")
    else:
        country_names.append(cName)
        coords = aCountry['geometry']['coordinates'][0]
        country_xs.append(list(map(lambda x:x[0],coords)))
        country_ys.append(list(map(lambda x:x[1],coords)))
        country_colors.append("green")
# purple
# green
source = ColumnDataSource(
    data = dict(
        x=country_xs,
        y=country_ys,
        color=country_colors,
        name=country_names
    )
)
# Plot the results
TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
p = figure(title="Countries with a Continous Boarder in Red", tools=TOOLS,
          tooltips=[
        ("Name","@name"),("(Long, Lat)", "($x, $y)")
    ], plot_width=600, plot_height=400)

p.patches('x', 'y',
    fill_color='color', fill_alpha=0.7,
    line_color="white", line_width=0.5,
    source=source)

hover = p.select(dict(type=HoverTool))
hover.point_policy = "follow_mouse"
show(p)