# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")
data.info()
from plotly.tools import FigureFactory as ff

datahead=data.head(10)

datahead=datahead.rename(index=str, columns={"battery_power": "Battery Power", "blue": "Bluetooth","clock_speed":"Clock Speed","dual_sim":"Dual Sim","fc":"Front Camera MP",

                                   "four_g":"4G","int_memory":"Internal Memory(GB)","m_dep":"Mobile Depth(CM)","mobile_wt":"Weight","n_cores":"Number of cores","pc":"Primary Camera MP"

                                   ,"px_height":"Pixel R. Height","px_width":"Pixel R. Width","ram":"RAM(MB)","sc_h":"Screen Height(cm)","sc_w":"Screen Width",

                                   "talk_time":"Longest Battery Charge","three_g":"3G","touch_screen":"Touch Screen","wifi":"WIFI","price_range":"Price Range"})

colorscale = "Greens"

table = ff.create_table(datahead,colorscale=colorscale,height_constant=40)

table.layout.width=2500

for i in range(len(table.layout.annotations)):

    table.layout.annotations[i].font.size = 8

iplot(table)
import missingno as msno

import matplotlib.pyplot as plt

msno.bar(data)

plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

corr=data.corr()

fig = plt.figure(figsize=(15,12))

r = sns.heatmap(corr, cmap='Purples')

r.set_title("Correlation ")
#price range correlation

corr.sort_values(by=["price_range"],ascending=False).iloc[0].sort_values(ascending=False)
import csv

import json

import re

import numpy as np

import pandas as pd

import altair as alt



from collections import Counter, OrderedDict

from IPython.display import HTML

from  altair.vega import v3



# The below is great for working but if you publish it, no charts show up.

# The workaround in the next cell deals with this.

alt.renderers.enable('notebook')



HTML("This code block contains import statements and setup.")
import altair as alt

from  altair.vega import v3

import json

alt.renderers.enable('notebook')

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

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

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

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

    workaround.format(json.dumps(paths)),

    "</script>",

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))
chart=alt.Chart(data).mark_bar().encode(

    alt.X('ram', bin=True),

    y='count(*):Q',

    color='price_range:N',

).facet(column='price_range:N')

render(chart)
chart=alt.Chart(data).mark_circle(size=20).encode(

    x='ram',

    y='battery_power',

    color='price_range:N',

    tooltip=["price_range", "ram",'battery_power']

).interactive().properties(

    width=400, height=300

)

render(chart)
nearest = alt.selection(type='single', nearest=True, on='mouseover',

                        fields=['ram'], empty='none')



line = alt.Chart(data).mark_line(interpolate='basis').encode(

    x='ram:Q',

    y='int_memory:Q',

    color='price_range:N'

)



selectors = alt.Chart(data).mark_point().encode(

    x='ram:Q',

    opacity=alt.value(0),

).add_selection(

    nearest

)





points = line.mark_point().encode(

    opacity=alt.condition(nearest, alt.value(1), alt.value(0))

)





text = line.mark_text(align='left', dx=5, dy=-5).encode(

    text=alt.condition(nearest, 'int_memory:Q', alt.value(' '))

)





rules = alt.Chart(data).mark_rule(color='black').encode(

    x='ram:Q',

).transform_filter(

    nearest

)





alt.layer(

    line, selectors, points, rules, text

).properties(

    width=400, height=300

)



render(line+selectors+points+text+rules)
f, ax = plt.subplots()

sns.despine(bottom=True, left=True)



# Show each observation with a scatterplot

sns.stripplot(x="touch_screen", y="ram", hue="price_range",

              data=data, dodge=True, jitter=True,

              alpha=.25, zorder=1)



# Show the conditional means



# Improve the legend 

handles, labels = ax.get_legend_handles_labels()

ax.legend( title="Price Range",

          handletextpad=0, columnspacing=1,

          loc="best", ncol=2, frameon=True)
zero=data[data.price_range==0]

one=data[data.price_range==1]

two=data[data.price_range==2]

three=data[data.price_range==3]



trace0 = go.Box(

    y=zero.ram.values,

    x=zero.three_g.values,

    name='0',

    marker=dict(

        color='#3D9970'

    )

)

trace1 = go.Box(

    y=one.ram.values,

    x=one.three_g.values,

    name='1',

    marker=dict(

        color='#FF4136'

    )

)

trace2 = go.Box(

    y=two.ram.values,

    x=two.three_g.values,

    name='2',

    marker=dict(

        color='#FF851B'

    )

)

trace3 = go.Box(

    y=three.ram.values,

    x=three.three_g.values,

    name='3',

    marker=dict(

        color='blue'

    )

)



data1 = [trace0, trace1, trace2,trace3]

layout = go.Layout(

    xaxis=dict(title="Three-g"),

    yaxis=dict(

        title="ram",

        zeroline=False

    ),

    boxmode='group'

)

fig = go.Figure(data=data1, layout=layout)

iplot(fig)
f, ax = plt.subplots(figsize=(10, 10))

ax=sns.swarmplot(x="four_g", y="ram", hue="price_range",

              palette="Dark2", data=data)

ax=sns.set(style="darkgrid")
chart=alt.Chart(data).mark_line(

    font="Helvetica Neue").encode(

    x='ram:Q',

    y='n_cores:O',

    color=alt.Color('price_range', scale=alt.Scale(scheme="Viridis")),

    tooltip=["price_range", "ram",'n_cores']

).properties(

    width=400,

    height=300

)

render(chart)
chart=alt.Chart(data).mark_circle(

    opacity=0.8,

    stroke='price_range',

    strokeWidth=1,

    strokeCap='square',

    shape="diamond"

).encode(

    alt.X('ram:O', axis=alt.Axis(labelAngle=0)),

    alt.Y('pc:N'),

    alt.Size('price_range:Q',

        scale=alt.Scale(range=[0, 200])

    ),

    alt.Color('price_range', scale=alt.Scale(scheme="plasma"))

).properties(

    width=400,

    height=300

)

render(chart)
bars = alt.Chart(data).mark_bar().encode(

    x=alt.X('mean(ram):Q'),

    y=alt.Y('fc:N'),

    color=alt.Color('price_range', scale=alt.Scale(scheme="viridis"))

    

)



text = alt.Chart(data).mark_text(dy=3, dx=-16, color='white').encode(

    x=alt.X('mean(ram):Q', stack='zero'),

    y=alt.Y('fc:N'),

    detail='price_range:N',

    text=alt.Text('mean(ram):Q', format='.0f')

)



render(bars + text)
chart=alt.Chart(data).mark_area().encode(

    x='ram:Q',

    y='mobile_wt:Q',

    color=alt.Color('price_range', scale=alt.Scale(scheme="magma")),

    row=alt.Row('price_range:N')

).properties(height=50, width=400)



render(chart)
chart=alt.Chart(data).mark_circle().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y(alt.repeat("row"), type='quantitative'),

    color=alt.Color('price_range', scale=alt.Scale(scheme="plasma"))

).properties(

    width=300,

    height=200

).repeat(

    column=["m_dep",'px_height',  'px_width','clock_speed'],

    row=["ram"]

).interactive()



render(chart)
chart=alt.Chart(data).mark_line(interpolate='step-after').encode(

    x='ram',

    y='talk_time',

    column="price_range",

    color=alt.Color('price_range', scale=alt.Scale(scheme="inferno"))

    

)

render(chart)
trace0 = go.Violin(

    y=zero.ram.values,

    x=zero.wifi.values,

    name='0',

    marker=dict(

        color='lightgreen'

    )

)

trace1 = go.Violin(

    y=one.ram.values,

    x=one.wifi.values,

    name='1',

    marker=dict(

        color='royalblue'

    )

)

trace2 = go.Violin(

    y=two.ram.values,

    x=two.wifi.values,

    name='2',

    marker=dict(

        color='mediumorchid'

    )

)

trace3 = go.Violin(

    y=three.ram.values,

    x=three.wifi.values,

    name='3',

    marker=dict(

        color='coral'

    )

)



data1 = [trace0, trace1, trace2,trace3]

layout = go.Layout(

    xaxis=dict(title="Wi-Fi"),

    yaxis=dict(

        title="ram",

        zeroline=False

    ),

    boxmode='group'

)

fig = go.Figure(data=data1, layout=layout)

iplot(fig)
g = sns.FacetGrid(data, col="dual_sim", hue="price_range", palette="Set1",height=5

                   )

g = (g.map(sns.distplot, "ram").add_legend())

color_scale = alt.Scale(domain=['0', '1',"2","3"],

                        range=['rgb(165,0,38)', 'rgb(253,174,97)','rgb(224,243,248)','rgb(49,54,149)'])

base = alt.Chart(data).mark_point().encode(

    x='ram',

     color=alt.Color('price_range', scale=color_scale)

).add_selection(

  

).interactive().properties(

    width=400,

    height=300,

    

)



render(base.encode(y='sc_h') | base.encode(y='sc_w'))
y = data["price_range"].values

x_data=data.drop(["price_range"],axis=1)

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)
from yellowbrick.target import ClassBalance

visualizer = ClassBalance(labels=[0, 1, 2,3])

visualizer.fit(y_train, y_test)

visualizer.poof()
from sklearn.svm import SVC

svm=SVC(random_state=1)

svm.fit(x_train,y_train)

print("train accuracy:",svm.score(x_train,y_train))

print("test accuracy:",svm.score(x_test,y_test))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

accuracy_list_train = []

k=np.arange(1,21,1)

for each in k:

    x_new = SelectKBest(f_classif, k=each).fit_transform(x_train, y_train)

    svm.fit(x_new,y_train)

    accuracy_list_train.append(svm.score(x_new,y_train))   

    

plt.plot(k,accuracy_list_train,color="green",label="train")

plt.xlabel("k values")

plt.ylabel("train accuracy")

plt.legend()

plt.show()



d = {'best features number': k, 'train_score': accuracy_list_train}

df = pd.DataFrame(data=d)

print("max accuracy:",df["train_score"].max())

print("max accuracy id:",df["train_score"].idxmax())
print(" max accuracy values: \n", df.iloc[4])
selector = SelectKBest(f_classif, k = 5)

x_new = selector.fit_transform(x_train, y_train)

x_new_test=selector.fit_transform(x_test,y_test)

names_train = x_train.columns.values[selector.get_support()]

names_test = x_test.columns.values[selector.get_support()]

print("x train features:",names_train)

print("x test features:",names_test)
from sklearn.model_selection import GridSearchCV



C=[1,0.1,0.25,0.5,2,0.75]

kernel=["linear","rbf"]

gamma=["auto",0.01,0.001,0.0001,1]

decision_function_shape=["ovo","ovr"]

svm=SVC(random_state=1)

grid_svm=GridSearchCV(estimator=svm,cv=5,param_grid=dict(kernel=kernel,C=C, gamma=gamma, decision_function_shape=decision_function_shape))

grid_svm.fit(x_new,y_train)

print("best score: ", grid_svm.best_score_)

print("best param: ", grid_svm.best_params_)


from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores

_, ax = plt.subplots()



# Create a cross-validation strategy

cv = StratifiedKFold(10)



# Create the cv score visualizer

oz = CVScores(

    SVC(C=2,decision_function_shape="ovo",gamma="auto",kernel="linear",random_state=1), ax=ax, cv=cv, scoring='accuracy'

)

oz.fit(x_new, y_train)

oz.poof()

svm_model=SVC(C=2,decision_function_shape="ovo",gamma="auto",kernel="linear",random_state=1)

svm_model.fit(x_new,y_train)
print("train_accuracy:",svm_model.score(x_new,y_train))

print("test_accuracy: ", svm_model.score(x_new_test,y_test))
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(

    svm_model, classes=[0,1,2,3]

)



cm.fit(x_new, y_train)

cm.score(x_new_test, y_test)



cm.poof()
y_pred=svm_model.predict(x_new_test)
svm_test=x_test[["battery_power","int_memory","px_height","px_width","ram"]]
svm_test["y_true"]=y_test

svm_test["y_pred"]=y_pred
svm_test.head()
brush = alt.selection(type='interval')

color_scale = alt.Scale(domain=['0', '1',"2","3"],

                        range=['black', 'green','red','purple'])

points = alt.Chart(title="Y True").mark_point(stroke='price_range',

    strokeWidth=4).encode(

 

    x='ram:Q',

    y='battery_power:Q',

    color=alt.condition(brush, 'y_true:N', alt.value('lightgray'),scale=color_scale)

).properties(

    selection=brush,

    width=800

)

points2=alt.Chart(title="Y Prediction").mark_point(stroke='price_range',

    strokeWidth=4).encode(

    x='ram:Q',

    y='battery_power:Q',

    color=alt.condition(brush, 'y_pred:N', alt.value('lightgray'))

).properties(

    selection=brush,

    width=800

)

# the bottom bar plot

bars = alt.Chart().mark_bar().encode(

    y='y_true:N',

    color='y_true:N',

    x='count(y_true):Q'

).transform_filter(

    brush.ref() # the filter transform uses the selection

                # to filter the input data to this chart

)

bar2 = alt.Chart().mark_bar().encode(

    y='y_pred:N',

    color='y_pred:N',

    x='count(y_pred):Q'

).transform_filter(

    brush.ref() # the filter transform uses the selection

                # to filter the input data to this chart

)

chart = alt.vconcat(points, bars,points2,bar2, data=svm_test)

render(chart)