# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import os

import bokeh.io

# from bokeh.layouts import row, column

from bokeh.models import ColumnDataSource, Label

from bokeh.plotting import figure, show



import umap

from sklearn.cluster import DBSCAN

from sklearn.feature_extraction.text import TfidfVectorizer

import ast

import re



bokeh.io.output_notebook()



# Any results you write to the current directory are saved as output.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
r_rcp = pd.read_csv('/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv', delimiter=',')

ingrdt = r_rcp.loc[:, ['steps']]

ing_all = r_rcp.loc[:, ['ingredients']]

food_data = ingrdt.values.tolist()

all_them = ing_all.values.tolist()



only_chicken = []

pos = 0

for ig in all_them:

    if 'chicken' in ig[0]:

        only_chicken.append(pos)

    pos+=1                  

print("CHICKEN {}".format(len(only_chicken)))

#raise ValueError("look")

list_stps = []



for f in range(len(food_data)-1):

    stps = ast.literal_eval(food_data[f][0])

    stps_stgr = ' '.join(stps)

    stps_clean = re.sub("[^a-zA-Z]", " ", stps_stgr)

    list_stps.append(stps_clean)



stps_chicken = [list_stps[oc] for oc in only_chicken]

print(stps_chicken[0:50])      
vectorizer = TfidfVectorizer(min_df=10,max_features=None)                                         

vz = vectorizer.fit_transform(stps_chicken)

print("VZ done")



clusterable_embedding = umap.UMAP(

    n_neighbors=30,  

    min_dist=0.0,

    n_components=2,

    random_state=42,

).fit_transform(vz)

print("umap done")



labels = DBSCAN(

    eps=0.18,

    min_samples=30).fit_predict(clusterable_embedding)

print("dbscan done")



clustered = (labels >= 0)

xtx = clusterable_embedding[clustered, 0]

ytx = clusterable_embedding[clustered, 1]



xtx_n = clusterable_embedding[~clustered, 0]

ytx_n = clusterable_embedding[~clustered, 1]



n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters {}".format(n_clusters_))
llbl = list(labels[clustered])

color_YlOrRd = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a#','e31a1c','#bd0026','#800026']

col = [color_YlOrRd[i] if (i>=0 and i < 9) else '#9119e6' for i in llbl]
sourcetx = ColumnDataSource(data=dict(xtx=xtx, ytx=ytx, col=col))

source_noise = ColumnDataSource(data=dict(xtx_n=xtx_n, ytx_n=ytx_n))



ptx = figure(plot_width=800, plot_height=600,

             title="Ingredients: Python, umap, Dbscan, Bokeh.",

             tools="pan,wheel_zoom,reset",

             active_scroll="wheel_zoom",

             toolbar_location="above"

             )
# output_file("recipe.html")



ptx.scatter('xtx', 'ytx', size=3, alpha=0.8, line_dash='solid', color="col", source=sourcetx,

                     legend='clustered')

ptx.scatter('xtx_n', 'ytx_n', size=3, alpha=0.8, line_dash='solid', color='#CDCDCD',

                         source=source_noise, legend='noise') 

ptx.legend.click_policy = "hide"

ptx.legend.background_fill_alpha = 0.4

ptx.legend.location = "bottom_right"



show(ptx)