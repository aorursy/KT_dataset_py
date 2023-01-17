import pandas as pd

import numpy as np

import random

from sklearn import metrics

import seaborn as sns;sns.set()

import matplotlib.pyplot as plt

%matplotlib inline

import holoviews as hv

from holoviews import opts, dim, Palette

import geoviews as gv

hv.extension('bokeh', 'matplotlib')

opts.defaults(

    opts.Bars(xrotation=45, tools=['hover']),

    opts.Curve(width=600,height=400, tools=['hover']),

    

    opts.Scatter(width=800, height=600, color=Palette('Category20'), tools=['hover']),

    opts.NdOverlay(legend_position='top_left'))

df = pd.read_csv('../input/master.csv')
df.head()
df.shape
df.info()
df.describe()
df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

df_country["suicides_per_100k"]=df_country["suicides_no"]/(df_country["population"]/100000)

df_country["Average_suicide"] = (df_country["suicides_no"]/31).astype(int)

df_country.head()

def avg(): 

    g=df_country.sort_values("Average_suicide").tail(10)

    return(g)

   

avg()
def map():

    import geopandas as gpd

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    world=world.rename(index=str, columns={"name": "country"})

    

    df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

    df_country["suicides_per_100k"]=df_country["suicides_no"]/(df_country["population"]/100000)

    df_country["Average_suicide"] = (df_country["suicides_no"]/31).astype(int)

    df_country["country"]=df_country["country"].str.replace("Russian Federation","Russia")

    

    df1 = pd.merge(world, df_country,  on='country', how='outer')

    

    df1 = df1.dropna(axis=0)

    polys = gv.Polygons(df1, vdims=['suicides_per_100k',"Average_suicide", 'suicides_no', 'country'])

    polys.opts(width=800, height=400, tools=['hover'], cmap='viridis', ylim=(-60, 90))

    return(polys)

map()
def top_countries():

    df_country = df.groupby(["country","year"],as_index=False)["population","suicides_no"].sum()

    stage = df_country.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]



    macro = hv.Dataset(stage, ['country', 'year'])

    plot = macro.to(hv.Curve, 'year', ['suicides_no',"population"]).overlay()



    plot.relabel('Suicide Rate between 1985 - 2016')

    return(plot)

top_countries()
def all_countries():

    df_country=df.groupby(["country","year"],as_index=False)["population","suicides_no"].sum()

    macro = hv.Dataset(df_country, ['country', 'year'])

    curves = macro.to(hv.Curve, 'year', 'suicides_no', groupby='country')

    

    return(curves)

all_countries()

def suicides_per_100k():

    df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

    df_country["suicides_per_100k"]=df_country["suicides_no"]/(df_country["population"]/100000)

    plt.figure(figsize=(10,20))

    plt.axvline(df_country["suicides_per_100k"].mean(), color='r', linestyle='--')

    df_country.sort_values('suicides_per_100k',inplace=True)

    sns.barplot(y=df_country["country"], x=df_country["suicides_per_100k"])

    plt.xlabel('No. of suicides per 100k')

    plt.ylabel('Countries')

    plt.title('Total No. of Suicides per country from 1985 to 2016')

    return(plt.show())

suicides_per_100k()
def suicides_per_100ks():

    df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

    df_country["suicides_per_100k"]=df_country["suicides_no"]/(df_country["population"]/100000)

    macro = hv.Dataset(df_country, ["country"])

    bars = macro.to(hv.Bars, "country", 'suicides_per_100k')

    bars.opts(width=900,height=400)



    return(bars)

#suicides_per_100ks()
def max_suicide():

    df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

    plt.figure(figsize=(10,20))

    plt.axvline(df_country["suicides_no"].mean(), color='r', linestyle='--')

    df_country.sort_values('suicides_no',inplace=True)

    sns.barplot(y=df_country["country"], x=df_country["suicides_no"])

    plt.xlabel('No. of suicides')

    plt.ylabel('Countries')

    plt.title('Total No. of Suicides per country from 1985 to 2016')

    return(plt.show())

max_suicide()
def max_suicides():

    df_country=df.groupby(["country"],as_index=False)["population","suicides_no"].sum()

    macro = hv.Dataset(df_country, ["country"])

    bars = macro.to(hv.Bars, "country", 'suicides_no')

    bars.opts(width=900,height=400)



    return(bars)

   

#max_suicides()
def past_ten_years():

    df_country=df.groupby(["country","year","sex"],as_index=False)["population","suicides_no"].sum()

   # df_country = df_country[df_country["year"] > 2005]

    stage = df_country.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]



    macro = hv.Dataset(stage, ['year',"sex"])

    bars = macro.sort('country').to(hv.Bars, 'country', 'suicides_no')

    bars.opts(width=600,height=400)



    return(bars)

past_ten_years()
def gender():

    df_country=df.groupby(["country","year","sex"],as_index=False)["population","suicides_no"].sum()

    df_country = df_country[df_country["year"] > 2005]

    stage = df_country.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]

    stagem = stage.loc[stage['sex']=="male"]

    stagef = stage.loc[stage['sex']=="female"]

    

    macrof = hv.Dataset(stagef, ['country', 'year'])

    macrom = hv.Dataset(stagem, ['country', 'year'])

    curvem = macrom.to(hv.Curve, 'year', ['suicides_no', 'sex'], label="Male")

    curvef = macrof.to(hv.Curve, 'year', ['suicides_no', 'sex'], label="Female")

    

    curves=(curvem * curvef)

    

    return(curves)

gender()
def generations():

    df_country = df.groupby(["country","year","generation"],as_index=False)["population","suicides_no"].sum()

    df_s = df_country[df_country["year"] > 2005]

    

    stage = df_s.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]



    stage1 = stage.loc[stage['generation'].str.contains("Generation X")]

    stage2 = stage.loc[stage['generation'].str.contains("Silent")]

    stage3 = stage.loc[stage['generation'].str.contains("G.I. Generation")]

    stage4 = stage.loc[stage['generation'].str.contains("Boomers")]

    stage5 = stage.loc[stage['generation'].str.contains("Millenials")]

    stage6 = stage.loc[stage['generation'].str.contains("Generation Z")]

    

    

    macro1 = hv.Dataset(stage1, ['country', 'year'])

    macro2 = hv.Dataset(stage2, ['country', 'year'])

    macro3 = hv.Dataset(stage3, ['country', 'year'])

    macro4 = hv.Dataset(stage4, ['country', 'year'])

    macro5 = hv.Dataset(stage5, ['country', 'year'])

    macro6 = hv.Dataset(stage6, ['country', 'year'])

    

    

    curve1 = macro1.to(hv.Curve, 'year', 'suicides_no', label="Generation X")

    curve2 = macro2.to(hv.Curve, 'year', 'suicides_no', label="Silent")

    curve3 = macro3.to(hv.Curve, 'year', 'suicides_no', label="G.I. Generation")

    curve4 = macro4.to(hv.Curve, 'year', 'suicides_no', label="Boomers")

    curve5 = macro5.to(hv.Curve, 'year', 'suicides_no', label="Millenials")

    curve6 = macro6.to(hv.Curve, 'year', 'suicides_no', label="Generation Z")

    

    

    curves=(curve1 * curve2 * curve3 *curve4 *curve5 *curve6).opts( legend_position='top_right')

    

    

    return(curves)

generations()
def age():

    df_country=df.groupby(["country","year","age"],as_index=False)["population","suicides_no"].sum()

    df_s=df_country[df_country["year"] > 2005]

    stage = df_country.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]

    

    

    macro = hv.Dataset(stage, ['year',"age"])

    bars = macro.sort('country').to(hv.Bars, 'country', 'suicides_no')

    bars.opts(width=600,height=400)

    return(bars)

age()
def ages():

    df_country = df.groupby(["country","year","age"],as_index=False)["population","suicides_no"].sum()

   # df_s = df_country[df_country["year"] > 2005]

    stage = df_country.loc[df_country['country'].isin(["Mexico","Republic of Korea","France","Japan","Russian Federation","United States","Brazil"])]

    stage1 = stage.loc[stage['age'].str.contains("15-24 years")]

    stage2 = stage.loc[stage['age'].str.contains("35-54 years")]

    stage3 = stage.loc[stage['age'].str.contains("75+ years")]

    stage4 = stage.loc[stage['age'].str.contains("25-34 years")]

    stage5 = stage.loc[stage['age'].str.contains("55-74 years")]

    stage6 = stage.loc[stage['age'].str.contains("5-14 years")]

    

    

    macro1 = hv.Dataset(stage1, ['country', 'year'])

    macro2 = hv.Dataset(stage2, ['country', 'year'])

    macro3 = hv.Dataset(stage3, ['country', 'year'])

    macro4 = hv.Dataset(stage4, ['country', 'year'])

    macro5 = hv.Dataset(stage5, ['country', 'year'])

    macro6 = hv.Dataset(stage6, ['country', 'year'])

    

    

    curve1 = macro1.to(hv.Curve, 'year', 'suicides_no', label="15-24 years")

    curve2 = macro2.to(hv.Curve, 'year', 'suicides_no', label="35-54 years")

    curve3 = macro3.to(hv.Curve, 'year', 'suicides_no', label="75+ years")

    curve4 = macro4.to(hv.Curve, 'year', 'suicides_no', label="25-34 years")

    curve5 = macro5.to(hv.Curve, 'year', 'suicides_no', label="55-74 years")

    curve6 = macro6.to(hv.Curve, 'year', 'suicides_no', label="5-14 years")

    

    

    curves=(curve1 * curve2 * curve3 *curve4 *curve5 *curve6).opts( legend_position='top_right')

    

    

    return(curves)

ages()
def overall():

    

    df_country=df.groupby(["year","sex"],as_index=False)["population","suicides_no"].sum()

    

    macro = hv.Dataset(df_country, ['year',"sex"])

    bars = macro.to(hv.Bars, "sex", 'suicides_no')

    bars.opts(width=350,height=400)



    return(bars)



overall()
def gdp():

    df_gdp = df.groupby(["country","year","gdp_per_capita ($)"],as_index=False)["population","suicides_no"].sum()

    df_gdp["suicides_per_100k"] = df_gdp["suicides_no"]/(df_country["population"]/100000)

    sns.pairplot(df_gdp, x_vars=['suicides_no'], y_vars=["gdp_per_capita ($)"], height=6, aspect=2, kind='reg')

    X = df_gdp["gdp_per_capita ($)"].values.reshape(-1,1)

    y = df_gdp["suicides_no"]

    from sklearn.linear_model import LinearRegression

    lm = LinearRegression()

    lm.fit(X,y)

    print(lm.intercept_)

    print(lm.coef_)

    print("Y = {}".format(lm.coef_),"X + {}".format(lm.intercept_))

gdp()    