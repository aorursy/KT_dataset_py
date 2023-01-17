import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



### Plotly

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.plotly as py

from plotly import tools

init_notebook_mode(connected=True)



# Altair

import altair as alt



### Removes warnings that occassionally show up

import warnings

warnings.filterwarnings('ignore')
import json  # need it for json.dumps

from IPython.display import HTML



# Create the correct URLs for require.js to find the Javascript libraries

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + alt.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



altair_paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {paths}

}});

"""



# Define the function for rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        """Render an altair chart directly via javascript.

        

        This is a workaround for functioning export to HTML.

        (It probably messes up other ways to export.) It will

        cache and autoincrement the ID suffixed with a

        number (e.g. vega-chart-1) so you don't have to deal

        with that.

        """

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay defined and keep track of the unique div Ids

    return wrapped





@add_autoincrement

def render_alt(chart, id="vega-chart"):

    # This below is the javascript to make the chart directly using vegaEmbed

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vegaEmbed) {{

        const spec = {chart};     

        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

    }});

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(paths=json.dumps(altair_paths)),

    "</script>"

)))
DATA_PATH = "../input/"



train = pd.read_csv(DATA_PATH + "train.csv")

test  = pd.read_csv(DATA_PATH + "test.csv")



cat_columns = [col for col in train.columns if train[col].dtype == object]

print("Categorical columns:")

print(" --- ".join(cat_columns))



### Numerical columns

num_columns = [col for col in train.columns if train[col].dtype != object]

print("Numerical columns:")

print(" --- ".join(num_columns))

print()

print("Shape of train:", train.shape)

print("Shape of test:",  test.shape)
cover_type_map = dict()

cover_type_names = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",

                    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]



for ct_id, ct_name in zip(cover_type_names, range(1, 8)):

    cover_type_map[ct_name] = ct_id

    

train["Cover_Type"] = train["Cover_Type"].map(cover_type_map)

train["Cover_Type"].unique()
train_y = train["Cover_Type"]

train_id = train["Id"]

train_x = train.drop(["Cover_Type", "Id"], axis=1)



test_id = test["Id"]

test_x  = test.drop("Id", axis=1)



full    = pd.concat([train_x, test_x])

train_N = len(train_x)
temp = pd.DataFrame({"Dataset": ["Train", "Test"], "Number of Records": [train.shape[0], test.shape[0]]})



trace = go.Pie(labels=temp["Dataset"], values=temp["Number of Records"],

               hoverinfo='label+percent', textinfo='value', 

               textfont=dict(size=20),

               marker=dict(line=dict(color="black", width=2)))



layout = go.Layout(

    title = "Train/Test size difference",

)



fig = go.Figure(data=[trace], layout=layout)

iplot(fig);
def alt_soil_types(df, title=None):

    soil_types = [f"Soil_Type{i}" for i in range(1, 41)]

    

    temp = pd.DataFrame(df[soil_types].mean()).reset_index()

    temp.columns = ["Soil Type", "Fraction"]



    temp["Soil Type"] = temp["Soil Type"].map(lambda x: int(x.replace("Soil_Type", "")))

    temp = temp.sort_values("Soil Type")

    temp["Soil Type"] = temp["Soil Type"].map(lambda x: "Type " + str(x))

    temp["Fraction"]  = temp["Fraction"].round(5)



    chart = alt.Chart(temp).mark_bar().encode(

        alt.X("Soil Type:O", sort=None),

        alt.Y("Fraction:Q"),

        tooltip=["Soil Type", "Fraction"],

    ).properties(title=title)

    return chart



soil_types_train = alt_soil_types(train, title="Train Soil Type Distribution")

soil_types_test  = alt_soil_types(test, title="Test Soil Type Distribution")



chart = alt.vconcat(soil_types_train, soil_types_test)

render_alt(chart)
def alt_hist_chart(df, feat, df_name):    

    temp1 = np.histogram(df[feat], bins=100)

    temp1 = pd.DataFrame({feat: temp1[1][:-1].round(1), "Count": temp1[0]})



    bar = alt.Chart(temp1).mark_bar().encode(

        alt.X(feat, axis=alt.Axis(title=feat)),

        y = 'Count:Q',



        tooltip=[feat, "Count"]

    )



    temp2 = pd.DataFrame({"Mean": [round(df[feat].mean(), 1)]})

    line = alt.Chart(temp2).mark_rule(color='red').encode(

        alt.X("Mean:Q", axis=alt.Axis(title=feat)),

        size=alt.value(3),

        tooltip=["Mean"],

    )

    return (bar + line).properties(title=f"{feat} Histogram of {df_name} data").interactive()





slope_chart     = alt_hist_chart(train, "Slope", df_name="train")

elevation_chart = alt_hist_chart(train, "Elevation", df_name="train")

train_chart = alt.hconcat(slope_chart, elevation_chart)



slope_chart     = alt_hist_chart(test, "Slope", df_name="test")

elevation_chart = alt_hist_chart(test, "Elevation", df_name="test")

test_chart = alt.hconcat(slope_chart, elevation_chart)



chart = alt.vconcat(train_chart, test_chart)

render_alt(chart)
vc = train_y.value_counts()



trace = go.Bar(

    x=vc.index,

    y=vc.values,

)



layout = go.Layout(

    title   = "Cover Type shown in Train",

    xaxis   = dict(title = "Cover Type"),

    yaxis   = dict(title = "Count")

)



fig = go.Figure(data=[trace], layout=layout)

iplot(fig);
trace = go.Bar(

    x=["Lodgepole Pine", "Spruce/Fir", "Ponderosa Pine", "Krummholz", "Douglas-fir", "Aspen", "Cottonwood/Willow"],

    y=[253621, 217307, 34269, 21782, 18303, 18279, 2331],

)



layout = go.Layout(

    title   = "Predicted Cover Type (81% accurate)",

    xaxis   = dict(title = "Cover Type"),

    yaxis   = dict(title = "Count")

)



fig = go.Figure(data=[trace], layout=layout)

iplot(fig);
ignore_feats = [col for col in full.columns if "Soil_Type" in col]

corr         = full.drop(ignore_feats, axis=1).corr().round(3)



trace = go.Heatmap(

    x = corr.columns,

    y = corr.index,

    z = corr.values,

)



buttons = []



layout = dict(title = 'Correlation plots')



fig = dict(data=[trace], layout=layout)

iplot(fig)
temp = train[["Cover_Type", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"]].sample(1000)

temp["Cover_Type"] = temp["Cover_Type"].map(str)

temp.columns = ["Cover Type", "Horizontal Distance to Hydrology", "Vertical Distance to Hydrology"]



click = alt.selection_multi(fields=['Cover Type'])

palette = alt.Scale(domain=cover_type_names,

                    range=["purple", "#3498db", "#34495e", "#e74c3c", "teal", "#2ecc71", "olive"])





chart = (alt.Chart(temp)

 .mark_circle()

 .encode(

     x='Horizontal Distance to Hydrology',

     y='Vertical Distance to Hydrology',

     opacity=alt.value(0.5),

     tooltip=['Cover Type', "Horizontal Distance to Hydrology", "Vertical Distance to Hydrology"],

     color=alt.condition(click, 'Cover Type',

                         alt.value('lightgray'), scale=palette))

 .properties(selection=click, width=700, title="Cover Type")

 .interactive())



render_alt(chart)
plotly_data = []



colors = ["purple", "#3498db", "#34495e", "#e74c3c", "teal", "#2ecc71", "olive"]



for cover_type, color in zip(cover_type_names, colors):

    temp  = train[train["Cover_Type"] == cover_type]

    trace = go.Scatter3d(

        x = temp["Hillshade_9am"],

        y = temp["Hillshade_Noon"],

        z = temp["Hillshade_3pm"],

        mode = 'markers',

        name = cover_type,

        marker=dict(

            size=2,

            line=dict(

                color=color,

                width=0.5

            ),

            opacity=1

        )

    )

    plotly_data.append(trace)



layout = go.Layout(

    margin=dict(l=0, r=0, b=0),

    title="Hillshade distribution across target",

    scene=go.Scene(

        xaxis=go.XAxis(title='x - 9am'),

        yaxis=go.YAxis(title='y - Noon'),

        zaxis=go.ZAxis(title='z - 3pm')

    ),

    legend=dict(

        x=0,

        y=0.5,

        bordercolor='#000',

        borderwidth=1

    )

)



fig = go.Figure(data=plotly_data, layout=layout)

iplot(fig)
plotly_data = []



colors = ["purple", "#3498db", "#34495e", "#e74c3c", "teal", "#2ecc71", "olive"]



for cover_type, color in zip(cover_type_names, colors):

    temp  = train[train["Cover_Type"] == cover_type]

    trace = go.Scatter3d(

        x = temp["Elevation"],

        y = temp["Slope"],

        z = temp["Hillshade_9am"],

        mode   = 'markers',

        name   = cover_type,

        marker = dict(

            size = 2,

            line = dict(

                color = color,

                width = 0.5

            ),

            opacity = 1

        )

    )

    plotly_data.append(trace)



layout = go.Layout(

    margin = dict(l=0, r=0, b=0),

    title  = "Relationship between Shade, Slope, Elevation, and Cover Type",

    scene  = go.Scene(

        xaxis = go.XAxis(title="x - Elevation"),

        yaxis = go.YAxis(title="y - Slope"),

        zaxis = go.ZAxis(title="z - Hillshade 9am")

    ),

    legend = dict(

        x = 0,

        y = 0.5,

        bordercolor = '#000',

        borderwidth = 1

    )

)

fig = go.Figure(data=plotly_data, layout=layout)

iplot(fig)
feats = ["Wilderness_Area1", "Wilderness_Area2",  "Wilderness_Area3", "Wilderness_Area4"]



train["Wilderness_Area"] = ((1 * train["Wilderness_Area1"]) +

                            (2 * train["Wilderness_Area2"]) +

                            (3 * train["Wilderness_Area3"]) +

                            (4 * train["Wilderness_Area4"])).map(str)



vcs = pd.DataFrame()

for cover_type in cover_type_names:

    temp = train[train["Cover_Type"] == cover_type]

    vc   = temp["Wilderness_Area"].value_counts()

    size = len(temp)

    vc   = (vc / size).round(2)

    vc = pd.DataFrame(vc).T

    vc.index = [cover_type]

    vcs = pd.concat([vcs, vc])



plotly_data = []

for area in range(1, 5):

    vc = vcs[str(area)]

    trace = go.Bar(

        x=vc.index,

        y=vc.values,

        name=f"Area {area}"

    )

    plotly_data.append(trace)

    

layout = go.Layout(

    barmode = 'group',

    title   = "Wilderness Area",

    xaxis   = dict(title = "Cover Type"),

    yaxis   = dict(title = "Fraction")

)



fig = go.Figure(data=plotly_data, layout=layout)

iplot(fig);
feats = ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points", "Cover_Type"]

temp  = train[feats].sample(5000)



temp["Cover_Type"] = temp["Cover_Type"].map(str)

temp.columns = ["Distance to Roadways", "Distance to Fire Points", "Cover Type"]





click = alt.selection_multi(fields=['Cover Type'])

palette = alt.Scale(domain=cover_type_names,

                    range=["purple", "#3498db", "#34495e", "#e74c3c", "olive", "#2ecc71"])





chart = (alt.Chart(temp)

 .mark_square(size=8)

 .encode(

     x='Distance to Roadways',

     y='Distance to Fire Points',

     opacity=alt.value(0.7),

     tooltip=['Cover Type', "Distance to Roadways", "Distance to Fire Points"],

     color=alt.condition(click, 'Cover Type',

                         alt.value('lightgray'), scale=palette))

 .properties(selection=click, width=700, title="Roadways and Fire Points")

 .interactive())



render_alt(chart)
target = "Cover_Type"

feature = "Elevation"



fig = ff.create_distplot(

    [train[train[target] == y][feature].values for y in train[target].unique()], 

    train[target].unique(), 

    show_hist=False,

    show_rug=False,

)



for d in fig['data']:

    d.update({'fill' : 'tozeroy'})



layout = go.Layout(

    title   = "Elevation Distributions",

    xaxis   = dict(title = "Elevation (ft)"),

    yaxis   = dict(title = "Density"),

)



fig["layout"] = layout

iplot(fig)
def plotly_multi_dist(df, target=None, distributions=None, xlabel=None):

    assert target and distributions, "Please provide target and distribution variables"

    plotly_data = list()



    for dist in distributions:

        fig = ff.create_distplot(

            [df[df[target] == c][dist].values for c in df[target].unique()], 

            df[target].unique(), 

            show_hist=False,

            show_rug=False,

        )



        for d in fig['data']:

            d.update({'fill' : 'tozeroy'})



        plotly_data.append(fig["data"])



    plotly_data = [x for t in zip(*plotly_data) for x in t]



    buttons = list()

    for i, dist in enumerate(distributions):

        visibility = [i==j for j in range(len(distributions))]

        button = dict(

                     label  =  dist,

                     method = 'update',

                     args   = [{'visible': visibility}, {'title': dist}])

        buttons.append(button)



    updatemenus = list([

        dict(active  = -1,

             x       = -0.15,

             buttons = buttons

        )

    ])



    layout = dict(title="Distributions",

                  updatemenus=updatemenus,

                  xaxis   = dict(title = xlabel),

                  yaxis   = dict(title = "Density"))



    fig = dict(data=plotly_data, layout=layout)

    return fig



distributions = ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",

                 "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"]

fig = plotly_multi_dist(train, target="Cover_Type", distributions=distributions, xlabel="Distance (meters)")

iplot(fig)
distributions = ["Aspect", "Slope", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]

fig = plotly_multi_dist(train, target="Cover_Type", distributions=distributions)

iplot(fig)