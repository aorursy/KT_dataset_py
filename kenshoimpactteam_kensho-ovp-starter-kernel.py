import json

import os



import numpy as np

import pandas as pd 

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
base_path = "/kaggle/input/kensho-ohio-voter-project"

all_files = os.listdir(base_path)

geo_files = [file for file in all_files if file.endswith("geojson")]

demo_county_files = [file for file in all_files if file.startswith("oh_counties")]

demo_cousub_files = [file for file in all_files if file.startswith("oh_county_sub")]

demo_cd116_files = [file for file in all_files if file.startswith("oh_cong_dist")]



print("base_path=", base_path)

print()

print("geography files")

for file in geo_files:

    print("  ", file)

print()

print("demographic county files")

for file in demo_county_files:

    print("  ", file)

print()

print("demographic sub county files")

for file in demo_cousub_files:

    print("  ", file)

print()

print("demographic congressional district files")

for file in demo_cd116_files:

    print("  ", file)
def read_geojson(entity):

    ss = "us" if entity == "zcta510" else "39"

    file_path = os.path.join(base_path, f"cb_2019_{ss}_{entity}_500k.geojson")

    with open(file_path, "r") as fp:

        gjson = json.load(fp)

    return gjson
def df_from_geojson(gjson):

    df = pd.DataFrame({

        key: [feature["properties"][key] for feature in gjson["features"]]

        for key in gjson["features"][0]["properties"]

    })

    return df
gjsons = {

    "county": read_geojson("county"),

    "cousub": read_geojson("cousub"),

    "cd116": read_geojson("cd116"),

}
dfs_geo = {

    "county": df_from_geojson(gjsons["county"]),

    "cousub": df_from_geojson(gjsons["cousub"]),

    "cd116": df_from_geojson(gjsons["cd116"]),

}
dfs_geo["county"]
dfs_geo["cousub"]
dfs_geo["cd116"]
def generate_county_demo(df_geo):

    df_demo = df_geo[["COUNTYFP", "NAME"]].copy()

    quants = ["median_age", "median_income", "total_pop", "black_pop", "hisp_pop", "white_pop"]

    file_map = {

        "median_age": "oh_counties_median_age_2018.csv",

        "median_income": "oh_counties_median_income_2018.csv",

        "total_pop": "oh_counties_total_pop_2018.csv",

        "black_pop": "oh_counties_black_2018.csv",

        "hisp_pop": "oh_counties_hisp_2018.csv",

        "white_pop": "oh_counties_white_alone_not_hisp_2018.csv",

    }

        

    for qq in quants:

        file_path = os.path.join(base_path, "{}".format(file_map[qq]))

        df = pd.read_csv(file_path)[["county_fips", "value"]]

        df = df.rename(columns={"value": qq})

        df["COUNTYFP"] = df["county_fips"].apply(lambda x: '{:0>3d}'.format(x))

        df = df.drop(columns=["county_fips"])

        df_demo = pd.merge(df_demo, df, on="COUNTYFP")



    df_demo["black_perc"] = df_demo["black_pop"] / df_demo["total_pop"]

    df_demo["hisp_perc"] = df_demo["hisp_pop"] / df_demo["total_pop"]

    df_demo["white_perc"] = df_demo["white_pop"] / df_demo["total_pop"]

    return df_demo
def generate_cousub_demo(df_geo):

    df_demo = df_geo[["AFFGEOID", "NAME"]].copy()

    file_path = os.path.join(base_path, "oh_county_sub_perc_white_non_hisp.csv")

    df = pd.read_csv(file_path) 

    df["AFFGEOID"] = df["geoid"].apply(lambda x: x.replace("06000", "0600000"))

    df = df[["AFFGEOID", "total_pop", "white_pop", "perc_white"]]

    df = df.rename(columns={"perc_white": "white_perc"})

    df_demo = pd.merge(df_demo, df, on="AFFGEOID")

    return df_demo
def generate_cd116_demo(df_geo):

    df_demo = df_geo[["CD116FP"]].copy()

    quants = ["median_age", "median_income", "total_pop", "black_pop", "hisp_pop", "white_pop"]

    file_map = {

        "median_age": "oh_cong_dist_median_age_2018.csv",

        "median_income": "oh_cong_dist_median_income_2018.csv",

        "total_pop": "oh_cong_dist_total_pop_2018.csv",

        "black_pop": "oh_cong_dist_black_2018.csv",

        "hisp_pop": "oh_cong_dist_hisp_2018.csv",

        "white_pop": "oh_cong_dist_white_alone_not_hisp_2018.csv",

    }

        

    for qq in quants:

        file_path = os.path.join(base_path, "{}".format(file_map[qq]))

        df = pd.read_csv(file_path)[["cong_district", "value"]]

        df = df.rename(columns={"value": qq})

        df["CD116FP"] = df["cong_district"].apply(lambda x: '{:0>2d}'.format(x))

        df = df.drop(columns=["cong_district"])

        df_demo = pd.merge(df_demo, df, on="CD116FP")



    df_demo["black_perc"] = df_demo["black_pop"] / df_demo["total_pop"]

    df_demo["hisp_perc"] = df_demo["hisp_pop"] / df_demo["total_pop"]

    df_demo["white_perc"] = df_demo["white_pop"] / df_demo["total_pop"]

    return df_demo
dfs_demo = {}
dfs_demo["county"] = generate_county_demo(dfs_geo["county"])

dfs_demo["county"]
dfs_demo["cousub"] = generate_cousub_demo(dfs_geo["cousub"])

dfs_demo["cousub"]
dfs_demo["cd116"] = generate_cd116_demo(dfs_geo["cd116"])

dfs_demo["cd116"]
def plot_age_income(df_demo):

    plot_cols = ["median_age", "median_income"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=plot_cols)



    fig.append_trace(go.Histogram(x=df_demo["median_age"]), 1, 1)

    fig.append_trace(go.Histogram(x=df_demo["median_income"]), 1, 2)



    fig = fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        height=350,

        width=700,

        showlegend=False,

    )

    fig.show() 
plot_age_income(dfs_demo["county"])
plot_age_income(dfs_demo["cd116"])
def plot_pop(df_demo):

    plot_cols = ["total_pop", "black_pop", "hisp_pop", "white_pop"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=plot_cols)



    fig.append_trace(go.Histogram(x=df_demo["total_pop"]), 1, 1)

    fig.update_yaxes(type="log")

    fig.append_trace(go.Histogram(x=df_demo["black_pop"]), 1, 2)

    fig.update_yaxes(type="log")

    fig.append_trace(go.Histogram(x=df_demo["hisp_pop"]), 2, 1)

    fig.update_yaxes(type="log")

    fig.append_trace(go.Histogram(x=df_demo["white_pop"]), 2, 2)

    fig.update_yaxes(type="log")



    fig = fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        height=700,

        width=700,

        showlegend=False,

    )

    

    fig.show()
plot_pop(dfs_demo["county"])
plot_pop(dfs_demo["cd116"])
def plot_perc(df_demo):

    plot_cols = ["black_perc", "hisp_perc", "white_perc"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=plot_cols)

    xbins={"start": 0, "end": 1}

    

    fig.append_trace(go.Histogram(

        x=df_demo["black_perc"], xbins=xbins, bingroup=1), 1, 1)

#    fig.update_yaxes(type="log")

    fig.update_xaxes(range=(0,1))

    

    fig.append_trace(go.Histogram(

        x=df_demo["hisp_perc"], xbins=xbins, bingroup=1), 1, 2)

#    fig.update_yaxes(type="log")

    fig.update_xaxes(range=(0,1))

    

    fig.append_trace(go.Histogram(

        x=df_demo["white_perc"], xbins=xbins, bingroup=1), 2, 1)

#    fig.update_yaxes(type="log")

    fig.update_xaxes(range=(0,1))



    fig = fig.update_layout(

        margin={"r":0,"t":50,"l":0,"b":0},

        height=700,

        width=700,

        showlegend=False,

    )

    

    fig.show()
plot_perc(dfs_demo["county"])
plot_perc(dfs_demo["cd116"])
DEFAULT_QUANTS = [

    "median_age", "median_income", 

    "total_pop", "black_pop", "hisp_pop", "white_pop", 

    "black_perc", "hisp_perc", "white_perc"

]

def plot_choropleths(gjson, df_demo, entity, quants=DEFAULT_QUANTS):



    if entity == "county":

        hover_data = ["NAME"]

        locations="COUNTYFP"

        featureidkey="properties.COUNTYFP"

    elif entity == "cousub":

        hover_data = ["NAME"]

        locations="AFFGEOID"

        featureidkey="properties.AFFGEOID"

    elif entity == "cd116":

        hover_data = []

        locations="CD116FP"

        featureidkey="properties.CD116FP"

    

    

    for col in quants:

        range_color = (0.0, 1.0) if "perc" in col else None

    

        fig = px.choropleth(

            df_demo,

            geojson=gjson,

            locations=locations,

            featureidkey=featureidkey,

            color=col,

            color_continuous_scale="Viridis",

            range_color=range_color,

            hover_data=hover_data,

        )

        fig = fig.update_geos(

            visible=False,

            lonaxis={"range": (-85,-80.3)},

            lataxis={"range": (38.2,42.1)},

        )

        fig = fig.update_layout(

            margin={"r":0,"t":0,"l":0,"b":0},

            height=600,

        )

        fig.show()
plot_choropleths(gjsons["county"], dfs_demo["county"], "county")
plot_choropleths(gjsons["cousub"], dfs_demo["cousub"], "cousub", quants=["white_perc"])
plot_choropleths(gjsons["cd116"], dfs_demo["cd116"], "cd116")