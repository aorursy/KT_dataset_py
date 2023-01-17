import numpy as np 

import pandas as pd 

import plotly.express as px



import ipywidgets as widgets
country_codes = pd.read_csv("../input/iso-country-codes-global/wikipedia-iso-country-codes.csv")

country_codes = country_codes.set_index("Alpha-2 code")["English short name lower case"]

country_codes.name = "country"
df = pd.read_csv("../input/unemployment-in-european-union/une_rt_m.tsv", sep="\t")

cols = df.columns[0].split(",")  # Processed columns

df = df.merge(pd.DataFrame(dict(zip(cols, s.split(","))) for s in df.iloc[:, 0].values), left_index=True, right_index=True)

df = df.iloc[:, 1:]

df[r"country"] = df[r"geo\time"].map(country_codes)

df.loc[df["country"].isna(), "country"] = df.loc[df["country"].isna(), r"geo\time"]

df = df[df["s_adj"] == "SA"]

df.drop([r"geo\time", "s_adj"], axis=1, inplace=True)



# Assert no missing values 

assert not df.isna().any().any(), "Missing values found"



# Replace collons with nans

df.replace(r"\s*:\s*", np.nan, inplace=True, regex=True)



print(df.shape)

df.head()
ts_df = df.melt(id_vars=["age", "unit", "sex", "country"]).set_index("variable")

ts_df.index = pd.to_datetime([c.replace("M", " ") for c in ts_df.index])
ts_df["value"] = ts_df["value"].str.extract(r"(\d+\.*\d*)").astype(float)
ts_df_pct = ts_df[ts_df["unit"] == "PC_ACT"].drop("unit", axis=1)  # Percentage Active Population

ts_df_th = ts_df[ts_df["unit"] == "THS_PER"].drop("unit", axis=1)  # Thousands
def f(country):

    fig=px.line(ts_df_pct[ts_df_pct["country"] == country], y="value", color="sex", facet_col="age", title=f"Unemployment in {country}")

    return fig



w = widgets.interact(f, country=ts_df_pct["country"].unique())
px.box(ts_df_pct, x="age", color="sex", y="value", title="Unemployment Rate")
unemployment_covid_rise = [(k, max(v) - min(v)) for k, v in ts_df_pct.loc["2020":, :].set_index("value").groupby("age sex country".split()).groups.items()]

covid_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([v[0] for v in unemployment_covid_rise]), data=[v[1] for v in unemployment_covid_rise]).reset_index()

covid_df.columns = ["age", "sex", "country", "value"]

px.bar(covid_df.sort_values("value"), x="country", y="value", color="sex", facet_col="age", title="Difference between Maximum and Minimum Unemployment Rate per country in 2020")
px.box(covid_df, x="age", color="sex", y="value", title="Difference between Maximum and Minimum Unemployment by group in 2020")
min_year = ts_df_pct.index.year.min()

max_year = ts_df_pct.index.year.max()

w = widgets.IntRangeSlider(

    value=[2000, 2020],

    min=min_year,

    max=max_year,

    step=1,

    description='Select year:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d',

)



def f(x):

    title = f"Correlation of unemployment rates between the years {x[0]} and {x[1]}"

    fig = px.imshow(ts_df_pct.loc[str(x[1]):str(x[0]), :].reset_index().pivot(index=["index", "age", "sex"], columns="country", values="value").loc[(slice(None), "TOTAL", "T"), :].corr(), title=title, width=1000, height=1000)

    return fig



a = widgets.interact(f, x=w)