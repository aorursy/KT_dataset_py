from IPython.display import YouTubeVideo,HTML

YouTubeVideo("mkYBxfKDyv0", width=800, height=500)
import glob, pylab, pandas as pd

import pydicom, numpy as np

from os import listdir

from os.path import isfile, join

import matplotlib.pylab as plt

import matplotlib.pyplot as plt2

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go

from plotly import tools

import os

import seaborn as sns

from keras import layers

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *

from keras.applications import DenseNet121

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import tensorflow as tf

from plotly.tools import FigureFactory as FF

import plotly

from tqdm import tqdm

import cv2

from PIL import Image

from plotly.offline import iplot

import cufflinks

import cv2 as cv

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

%load_ext autoreload

plotly.io.templates.default = "none"

import altair as alt; import altair_render_script

plt.style.use("fivethirtyeight")

def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0, limit=100):

    cnt_srs = df[col].value_counts()[:limit]

    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h',

        marker=dict(color=color))



    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)

    data = [trace]

    annotations = []

    annotations += [go.layout.Annotation(x=673, y=100, xref="x", yref="y", text="(Most Popular)", showarrow=False, arrowhead=7, ax=0, ay=-40)]

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)

    

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
# Images Example

train_images_dir = '../input/siim-isic-melanoma-classification/train/'

train_images = [f for f in listdir(train_images_dir) if isfile(join(train_images_dir, f))]

test_images_dir = '../input/siim-isic-melanoma-classification/test/'

test_images = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]

print('5 Training images', train_images[:5]) # Print the first 5
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
fig=plt.figure(figsize=(15, 10))

columns = 5; rows = 4

for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(train_images_dir + train_images[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(-ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot
train.head()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
# Missing values statistics

missing_values = missing_values_table(train)

missing_values.head(20)
missing_values = missing_values_table(test)

missing_values.head(20)
train['age_approx2'] = train['age_approx'].fillna(0).astype('str')
holiday_json = {

"name": "flare",

"children": [

{

"name": "Affected Part",

"children":[

{

    "name": "Torso",

         "children": [

              {"name": "Chest", "size": 3.0},

              {"name": "Stomach (outer covering)", "size": 1.0},

              {"name": "Back of torso", "size": 2.0}

     ]

},

{

    "name": "Head/Neck",

         "children": [

              {"name": "Eyes", "size": 3.0},

              {"name": "Nose", "size": 1.0},

              {"name": "Neck", "size": 2.0},

              {"name": "Lips", "size": 1.0},

              {"name": "Ears", "size": 1.0}

     ]

},

{"name": "Upper extremity                                      ", "size": 3.0},

{"name": "Lower extremity                                      ", "size": 3.0},

{"name": "Palms/soles                                          ", "size": 2.0},

{"name": "Oral/genitalia                                       ", "size": 1.0}

]

},

{

"name":  "diagnosis",

"children":[

{"name": "unkown",       "size": 10.0},

{"name": "nevus",       "size": 4.0},

{"name": "melanoma",   "size": 3.0},

{"name": "seborheic keratosis",       "size": 2.0},

{"name": "lichenoid keratosis",         "size": 2.0},

{"name": "solar lentigo",                          "size": 1.0},

{"name": "atypical melanocytic proliferation",     "size": 1.0},

{"name": "cafe-au-lait macule", "size": 1.0}

]

},

{

"name": "Target feature",

"children": [

{"name": "malignant          ", "size": 1.0},

{"name": "benign                    ", "size": 6.0}

]

},

{

"name": "Age (approximated)",

"children":[

{

    "name": "0",

         "children": [

              {"name": "1", "size": 1.0},

              {"name": "2", "size": 1.0},

              {"name": "3", "size": 1.0},

              {"name": "4", "size": 1.0},

              {"name": "5", "size": 1.0}

     ]

},

{

    "name": "5",

         "children": [

              {"name": "6", "size": 1.0},

              {"name": "7", "size": 1.0},

              {"name": "8", "size": 1.0},

              {"name": "9", "size": 1.0},

              {"name": "10", "size": 1.0}

     ]

},

{

    "name": "20",

         "children": [

              {"name": "11", "size": 1.0},

              {"name": "12", "size": 1.0},

              {"name": "13", "size": 1.0},

              {"name": "14", "size": 1.0},

              {"name": "15", "size": 1.0},

              {"name": "16", "size": 1.0},

              {"name": "17", "size": 1.0},

              {"name": "18", "size": 2.0},

              {"name": "19", "size": 1.0},

              {"name": "20", "size": 1.0}

     ]

},

{

    "name": "25",

         "children": [

              {"name": "21", "size": 1.0},

              {"name": "22", "size": 1.0},

              {"name": "23", "size": 2.0},

              {"name": "24", "size": 1.0},

              {"name": "25", "size": 3.0}

     ]

},

{

    "name": "30",

         "children": [

              {"name": "26", "size": 2.0},

              {"name": "27", "size": 1.0},

              {"name": "28", "size": 2.0},

              {"name": "29", "size": 2.0},

              {"name": "30", "size": 1.0}

     ]

},

{

    "name": "40",

         "children": [

              {"name": "31", "size": 1.0},

              {"name": "32", "size": 3.0},

              {"name": "33", "size": 2.0},

              {"name": "34", "size": 1.0},

              {"name": "35", "size": 3.0},

              {"name": "36", "size": 2.0},

              {"name": "37", "size": 1.0},

              {"name": "38", "size": 2.0},

              {"name": "39", "size": 1.0},

              {"name": "40", "size": 4.0}

     ]

},

{

    "name": "45",

         "children": [

              {"name": "41", "size": 5.0},

              {"name": "42", "size": 4.0},

              {"name": "43", "size": 4.0},

              {"name": "44", "size": 5.0},

              {"name": "45", "size": 5.0}

     ]

},

{

    "name": "50",

         "children": [

              {"name": "46", "size": 4.0},

              {"name": "47", "size": 5.0},

              {"name": "48", "size": 4.0},

              {"name": "49", "size": 4.0},

              {"name": "50", "size": 3.0}

     ]

},

{

    "name": "55",

         "children": [

              {"name": "51", "size": 3.0},

              {"name": "52", "size": 3.0},

              {"name": "53", "size": 3.0},

              {"name": "54", "size": 2.0},

              {"name": "55", "size": 3.0}

     ]

},

{

    "name": "60",

         "children": [

              {"name": "56", "size": 3.0},

              {"name": "57", "size": 3.0},

              {"name": "58", "size": 2.0},

              {"name": "59", "size": 2.0},

              {"name": "60", "size": 4.0}

     ]

},

{

    "name": "65",

         "children": [

              {"name": "61", "size": 2.0},

              {"name": "62", "size": 3.0},

              {"name": "63", "size": 4.0},

              {"name": "64", "size": 3.0},

              {"name": "65", "size": 3.0}

     ]

},

{

    "name": "70",

         "children": [

              {"name": "66", "size": 3.0},

              {"name": "67", "size": 2.0},

              {"name": "68", "size": 2.0},

              {"name": "69", "size": 2.0},

              {"name": "70", "size": 2.0}

     ]

},

{

    "name": "75",

         "children": [

              {"name": "71", "size": 2.0},

              {"name": "72", "size": 2.0},

              {"name": "73", "size": 1.0},

              {"name": "74", "size": 1.0},

              {"name": "75", "size": 2.0}  

         ]

},

{

    "name": "80",

         "children": [

              {"name": "76", "size": 2.0},

              {"name": "77", "size": 2.0},

              {"name": "78", "size": 1.0},

              {"name": "79", "size": 1.0},

              {"name": "80", "size": 1.0}

     ]

},

{

    "name": "85",

         "children": [

              {"name": "81", "size": 2.0},

              {"name": "82", "size": 1.0},

              {"name": "83", "size": 1.0},

              {"name": "84", "size": 1.0},

              {"name": "85", "size": 1.0}

     ]

},

{

    "name": "90",

         "children": [

              {"name": "86", "size": 1.0},

              {"name": "87", "size": 1.0},

              {"name": "88", "size": 1.0},

              {"name": "89", "size": 1.0},

              {"name": "90", "size": 1.0},

     ]

},

]

}

] 

}               



import IPython

import json



with open('output.json', 'w') as outfile:  

    json.dump(holiday_json, outfile)

pd.read_json('output.json').head()



#Embedding the html string

html_string = """

<!DOCTYPE html>

<meta charset="utf-8">

<style>



.node {

  cursor: pointer;

}



.node:hover {

  stroke: #000;

  stroke-width: 1.5px;

}



.node--leaf {

  fill: white;

}



.label {

  font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;

  text-anchor: middle;

  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;

}



.label,

.node--root,

.node--leaf {

  pointer-events: none;

}



</style>

<svg width="760" height="760"></svg>

"""



js_string="""

 require.config({

    paths: {

        d3: "https://d3js.org/d3.v4.min"

     }

 });



  require(["d3"], function(d3) {



   console.log(d3);



var svg = d3.select("svg"),

    margin = 20,

    diameter = +svg.attr("width"),

    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");



var color = d3.scaleSequential(d3.interpolateViridis)

    .domain([-4, 4]);



var pack = d3.pack()

    .size([diameter - margin, diameter - margin])

    .padding(2);



d3.json("output.json", function(error, root) {

  if (error) throw error;



  root = d3.hierarchy(root)

      .sum(function(d) { return d.size; })

      .sort(function(a, b) { return b.value - a.value; });



  var focus = root,

      nodes = pack(root).descendants(),

      view;



  var circle = g.selectAll("circle")

    .data(nodes)

    .enter().append("circle")

      .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })

      .style("fill", function(d) { return d.children ? color(d.depth) : null; })

      .on("click", function(d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); });



  var text = g.selectAll("text")

    .data(nodes)

    .enter().append("text")

      .attr("class", "label")

      .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })

      .style("display", function(d) { return d.parent === root ? "inline" : "none"; })

      .text(function(d) { return d.data.name; });



  var node = g.selectAll("circle,text");



  svg

      .style("background", color(-1))

      .on("click", function() { zoom(root); });



  zoomTo([root.x, root.y, root.r * 2 + margin]);



  function zoom(d) {

    var focus0 = focus; focus = d;



    var transition = d3.transition()

        .duration(d3.event.altKey ? 7500 : 750)

        .tween("zoom", function(d) {

          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);

          return function(t) { zoomTo(i(t)); };

        });



    transition.selectAll("text")

      .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })

        .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })

        .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })

        .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });

  }



  function zoomTo(v) {

    var k = diameter / v[2]; view = v;

    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });

    circle.attr("r", function(d) { return d.r * k; });

  }

});

  });

 """

from IPython.core.display import display, HTML, Javascript

h2 = display(HTML("""<h2 style="font-family: 'Garamond';"> Zoomable Circle Packing </h2> <i>This is all primarily based on Anisotropic's work: upvote there, I admire his work a lot.</i>"""))

h = display(HTML(html_string))

j = IPython.display.Javascript(js_string)

IPython.display.display_javascript(j)



doc = {"name": "Characteristics", "color": "#ffae00", "percent": "", "value": "", "size": 25, "children": []}



def getsize(s):

    if s > 80:

        return 30

    elif s > 65:

        return 20

    elif s > 45:

        return 15

    elif s > 35:

        return 12

    elif s > 20:

        return 10 

    else:

        return 5

def vcs(col):

    vc = train[col].value_counts()

    keys = vc.index

    vals = vc.values 

    

    ddoc = {"name": col, "color": "#be5eff", "percent": "", "value": "", "size": 25, "children": []}

    for i,x in enumerate(keys):

        percent = round(100 * float(vals[i]) / sum(vals), 2)

        size = getsize(percent)

        collr = "#fc5858"

 

        doc = {"name": x+" ("+str(percent)+"%)", "color": collr, "percent": str(percent), "value": str(vals[i]), "size": size, "children": []}

        ddoc['children'].append(doc)

    return ddoc



# Coding Backgrounds

doc['children'].append(vcs('anatom_site_general_challenge'))

doc['children'].append(vcs('benign_malignant'))

doc['children'].append(vcs('diagnosis'))

doc['children'].append(vcs('sex'))



html_d1 = """<!DOCTYPE html><style>.node text {font: 12px sans-serif;}.link {fill: none;stroke: #ccc;stroke-width: 2px;}</style><svg id="four" width="760" height="900" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""

js_d1="""

require(["d3"], function(d3) {

var treeData = """ +json.dumps(doc) + """

var root, margin = {

        top: 20,

        right: 90,

        bottom: 120,

        left: 90

    },

    width = 960 - margin.left - margin.right,

    height = 660,

    svg = d3.select("#four").attr("width", width + margin.right + margin.left).attr("height", height + margin.top + margin.bottom).append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")"),

    i = 0,

    duration = 750,

    treemap = d3.tree().size([height, width]);



function collapse(t) {

    t.children && (t._children = t.children, t._children.forEach(collapse), t.children = null)

}



function update(n) {

    var t = treemap(root),

        r = t.descendants(),

        e = t.descendants().slice(1);

    r.forEach(function(t) {

        t.y = 180 * t.depth

    });

    var a = svg.selectAll("g.node").data(r, function(t) {

            return t.id || (t.id = ++i)

        }),

        o = a.enter().append("g").attr("class", "node").attr("transform", function(t) {

            return "translate(" + n.y0 + "," + n.x0 + ")"

        }).on("click", function(t) {

            t.children ? (t._children = t.children, t.children = null) : (t.children = t._children, t._children = null);

            update(t)

        });

    o.append("circle").attr("class", "node").attr("r", function(t) {

        return t.data.size

    }).style("fill", function(t) { return t.data.color;

    }), o.append("text").attr("dy", ".35em").attr("x", function( t) {

        return t.children || t._children ? -13 : 13

    }).attr("text-anchor", function(t) {

        return t.children || t._children ? "end" : "start"

    }).text(function(t) {

        return t.data.name

    });

    var c = o.merge(a);

    c.transition().duration(duration).attr("transform", function(t) {

        return "translate(" + t.y + "," + t.x + ")"

    }), c.select("circle.node").attr("r", function(t) {

        return t.data.size

    }).style("fill", function(t) {

        return t.data.color

    }).attr("cursor", "pointer");

    var l = a.exit().transition().duration(duration).attr("transform", function(t) {

        return "translate(" + n.y + "," + n.x + ")"

    }).remove();

    l.select("circle").attr("r", function(t) {

        return t.data.size

    }), l.select("text").style("fill-opacity", 1e-6);

    var d = svg.selectAll("path.link").data(e, function(t) {

        return t.id

    });

    console.log(), d.enter().insert("path", "g").attr("class", "link").attr("d", function(t) {

        var r = {

            x: n.x0,

            y: n.y0

        };

        return u(r, r)

    }).merge(d).transition().duration(duration).attr("d", function(t) {

        return u(t, t.parent)

    });

    d.exit().transition().duration(duration).attr("d", function(t) {

        var r = {

            x: n.x,

            y: n.y

        };

        return u(r, r)

    }).remove();



    function u(t, r) {

        var n = "M" + t.y + "," + t.x + "C" + (t.y + r.y) / 2 + "," + t.x + " " + (t.y + r.y) / 2 + "," + r.x + " " + r.y + "," + r.x;

        return console.log(n), n

    }

    r.forEach(function(t) {

        t.x0 = t.x, t.y0 = t.y

    })

}(root = d3.hierarchy(treeData, function(t) {

    return t.children

})).x0 = height / 2, root.y0 = 0, root.children.forEach(collapse), update(root);

});

"""

js7m="""require.config({

    paths: {

        d3: "https://d3js.org/d3.v4.min"

     }

 });

 

 require(["d3"], function(d3) {// Dimensions of sunburst.

 

 





var svg = d3.select("#fd"),

    width = +svg.attr("width"),

    height = +svg.attr("height");



var color = d3.scaleOrdinal(d3.schemeCategory20);



var simulation = d3.forceSimulation()



    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(120).strength(1))

    .force("charge", d3.forceManyBody().strength(-155))

    .force("center", d3.forceCenter(width / 2, height / 2));



d3.json("fd.json", function(error, graph) {

  if (error) throw error;



  var link = svg.append("g")

      .attr("class", "links")

    .selectAll("line")

    .data(graph.links)

    .enter().append("line")

      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });



// Define the div for the tooltip

var div = d3.select("body").append("div")	

    .attr("class", "tooltip")				

    .style("opacity", 0);



  var node = svg.append("g")

      .attr("class", "nodes")

    .selectAll("circle")

    .data(graph.nodes)

    .enter().append("circle")

      .attr("r", function(d) {return d.size})

      .attr("fill", function(d) { return color(d.group); })

      .call(d3.drag()

          .on("start", dragstarted)

          .on("drag", dragged)

          .on("end", dragended)).on("mouseover", function(d) {		

            div.transition()		

                .duration(200)		

                .style("opacity", .9);		

            div	.html(d.id )

                .style("left", (d3.event.pageX) + "px")		

                .style("top", (d3.event.pageY - 28) + "px");	

            })					

        .on("mouseout", function(d) {		

            div.transition()		

                .duration(500)		

                .style("opacity", 0);	

        });

          

    

// node.append("title")

  //  .text(function(d) { return d.id; });



  simulation

      .nodes(graph.nodes)

      .on("tick", ticked);

      



  simulation.force("link")

      .links(graph.links);



  function ticked() {

    link

        .attr("x1", function(d) { return d.source.x; })

        .attr("y1", function(d) { return d.source.y; })

        .attr("x2", function(d) { return d.target.x; })

        .attr("y2", function(d) { return d.target.y; });



    node

        .attr("cx", function(d) { return d.x; })

        .attr("cy", function(d) { return d.y; });

  }

});



function dragstarted(d) {

  if (!d3.event.active) simulation.alphaTarget(0.3).restart();

  d.fx = d.x;

  d.fy = d.y;

}



function dragged(d) {

  d.fx = d3.event.x;

  d.fy = d3.event.y;

}



function dragended(d) {

  if (!d3.event.active) simulation.alphaTarget(0);

  d.fx = null;

  d.fy = null;

}

 });

"""



h = display(HTML(html_d1))

j = IPython.display.Javascript(js_d1)

IPython.display.display_javascript(j)

ds = pydicom.dcmread(train_images_dir + train_images[0])

from skimage import measure

def plot_3d(image, threshold=-300):

    p = image.transpose(2,1,0)

    

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=1, allow_degenerate=True) 

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')



    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)



    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    plt.title("3-dimensional representation of an image")

    plt.show()



plot_3d(ds.pixel_array,threshold=100)
df = pd.DataFrame(train.benign_malignant.value_counts())

df['name'] = df.index

alt.Chart(df).mark_bar().encode(

    x='name',

    y='benign_malignant',

    tooltip=["name","benign_malignant"]

).interactive()
fig=plt.figure(figsize=(15, 10))

columns = 4; rows = 5

for i in range(1, columns*rows +1):

    # added the grid lines for pixel purposes

    ds = pydicom.dcmread(train_images_dir + train[train['benign_malignant']=='benign']['image_name'][i] + '.dcm')

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot
train.benign_malignant.value_counts()
# BY SERGEI ISSAEV

vals = train[train['benign_malignant']=='malignant']['image_name'].index.values

fig=plt.figure(figsize=(15, 10))

columns = 4; rows = 5

for i in range(1, columns*rows +1):

    # added the grid lines for pixel purposes



    ds = pydicom.dcmread(train_images_dir + train[train['benign_malignant']=='malignant']['image_name'][vals[i]] + '.dcm')

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    fig.add_subplot
df = pd.DataFrame(train.anatom_site_general_challenge.value_counts())

df['name'] = df.index

alt.Chart(df).mark_bar().encode(

    x='name',

    y='anatom_site_general_challenge',

    tooltip=["name","anatom_site_general_challenge"]

).interactive()
import squarify

fig = plt.figure(figsize=(25, 21))

marrimeko=train.anatom_site_general_challenge.value_counts().to_frame()

ax = fig.add_subplot(111, aspect="equal")

ax = squarify.plot(sizes=marrimeko['anatom_site_general_challenge'].values,label=marrimeko.index,

              color=sns.color_palette('cubehelix_r', 28), alpha=1)

ax.set_xticks([])

ax.set_yticks([])

fig=plt.gcf()

fig.set_size_inches(40,25)

plt.title("Treemap of cancer counts across different parts", fontsize=18)

plt.show();
_generate_bar_plot_hor(train, 'anatom_site_general_challenge', '<b>Affected Location</b>', '#66f992', 800, 400, 200)
def view_images(images, title = '', aug = None):

    width = 6

    height = 4

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')



    plt.subplots_adjust(wspace=0, hspace=0)

    plt.suptitle(title)

    plt.show()
%%time

for i in train['anatom_site_general_challenge'].unique()[:4]:

    view_images(train[train['anatom_site_general_challenge']==i]['image_name'], title=f"Growth in the {i}")
%%time

for i in train['anatom_site_general_challenge'][:1].unique():

    view_images(train[train['anatom_site_general_challenge']==i][train['target']==1]['image_name'], title=f"Malignant in the {i}");
_generate_bar_plot_hor(train, 'diagnosis', '<b>Affected Location</b>', '#f2b5bc', 800, 400, 200)
fig = plt.figure(figsize=(25, 21))

marrimeko=train.diagnosis.value_counts().to_frame()

ax = fig.add_subplot(111, aspect="equal")

ax = squarify.plot(sizes=marrimeko['diagnosis'].values,label=marrimeko.index,

              color=sns.color_palette('cubehelix_r', 28), alpha=1)

ax.set_xticks([])

ax.set_yticks([])

fig=plt.gcf()

fig.set_size_inches(40,25)

plt.title("Treemap of cancer counts across different ages", fontsize=18)

plt.show();
view_images(train[train['diagnosis']=='nevus']['image_name'], title="Nevus pigmentatious growth");
view_images(train[train['diagnosis']=='melanoma']['image_name'], title="Melanoma's growth");
train.diagnosis.value_counts()
def view_images_sp(images, title = '', aug = None):

    width = 1

    height = 1

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        plt.imshow(image, cmap=plt.cm.bone) 

        axs.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.suptitle(title)

    plt.show()

view_images(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo's growth");
view_images(train[train['diagnosis']=='lichenoid keratosis']['image_name'], title="Lichenoid's growth");
view_images(train[train['diagnosis']=='seborrheic keratosis']['image_name'], title="Lichenoid's growth");
view_images_sp(train[train['diagnosis']=='atypical melanocytic proliferation']['image_name'], title="Atypical melanocytic's growth");
df = pd.DataFrame(train.age_approx.value_counts())

df['name'] = df.index

alt.Chart(df).mark_bar().encode(

    x='name',

    y='age_approx',

    tooltip=["name","age_approx"]

).interactive()
fig= plt.figure(figsize=(22,6))

test["age_approx"].value_counts(normalize=True).to_frame().iplot(kind='bar',

                                                      yTitle='Percentage', 

                                                      linecolor='black', 

                                                      opacity=0.7,

                                                      color='red',

                                                      theme='pearl',

                                                      bargap=0.8,

                                                      gridcolor='white',                                                     

                                                      title='It does not exactly follow the same distribution in test though.')

plt.show()
import cv2

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        image = cv2.resize(image, (256, 256))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(256,256))

        image[:, :, 0] = clahe.apply(image[:, :, 0])

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
import matplotlib

matplotlib.font_manager._rebuild()

with plt.xkcd():



    fig = plt.figure()

    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))

    ax.spines['right'].set_color('none')

    ax.spines['top'].set_color('none')

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_ylim([-30, 10])



    data = np.ones(100)

    data[70:] -= np.arange(30)



    ax.annotate(

        'THE DAY I REALIZED\nI COULD USE APTOS METHODS\nIN OTHER COMPS',

        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))



    ax.plot(data)



    ax.set_xlabel('time')

    ax.set_ylabel('my overall sanity')

    fig.text(

        0.5, 0.05,

        '"My Mental Sanity Degrading over Time',

        ha='center')
import cv2

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (256, 256))

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image = cv2.resize(image, (256, 256))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)    

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(test_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        image = cv2.resize(image, (256, 256))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)    

    plt.suptitle(title)

view_images_aug(test[test['sex']=='male']['image_name'], title="Images for Male");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)    

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 



def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= circle_crop(image)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img

    

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= crop_image_from_gray(image)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
fgbg = cv.createBackgroundSubtractorMOG2()

    

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= fgbg.apply(image)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
fgbg = cv.createBackgroundSubtractorMOG2()

    

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        i = im // width

        j = im % width

        axs[i,j].imshow(thresh, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)

        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)



        # sure background area

        sure_bg = cv2.dilate(opening, kernel, iterations=3)



        # Finding sure foreground area

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)



        # Finding unknown region

        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        i = im // width

        j = im % width

        axs[i,j].imshow(sure_bg, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)

        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)



        # sure background area

        sure_bg = cv2.dilate(opening, kernel, iterations=3)



        # Finding sure foreground area

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)



        # Finding unknown region

        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)



        # Add one to all labels so that sure background is not 0, but 1

        markers = markers + 1



        # Now, mark the region of unknown with zero

        markers[unknown == 255] = 0

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)



        markers = cv2.watershed(image, markers)

        image[markers == -1] = [255, 0, 0]

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(image)

        fshift = np.fft.fftshift(f)

        magnitude_spectrum = 20*np.log(np.abs(fshift))



        i = im // width

        j = im % width

        axs[i,j].imshow(magnitude_spectrum, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
import albumentations as A

image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

albumentation_list = [A.RandomSunFlare(p=1), A.RandomFog(p=1), A.RandomBrightness(p=1),

                      A.RandomCrop(p=1,height = 512, width = 512), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","RandomFog","RandomBrightness",

               "RandomCrop","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

albumentation_list = [A.RandomSunFlare(p=1), A.GaussNoise(p=1), A.CLAHE(p=1),

                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",

               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),

                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",

               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        kernel = np.ones((5,5), np.uint8) 

  

        # The first parameter is the original image, 

        # kernel is the matrix with which image is  

        # convolved and third parameter is the number  

        # of iterations, which will determine how much  

        # you want to erode/dilate a given image.  

        img_erosion = cv2.erode(image, kernel, iterations=1) 

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        kernel = np.ones((5,5), np.uint8) 

    

        img_erosion = cv2.dilate(image, kernel, iterations=1) 

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        kernel = np.ones((5,5), np.uint8) 

  

        img_erosion = cv2.erode(image, kernel, iterations=1) 

        img_erosion = cv2.dilate(image, kernel, iterations=1) 

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");
image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (256, 256))

kernel = np.ones((5,5), np.uint8) 

  

image = cv2.erode(image, kernel, iterations=1) 

albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),

                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",

               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (256, 256))

kernel = np.ones((5,5), np.uint8) 

  

image = cv2.dilate(image, kernel, iterations=1) 

albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),

                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",

               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

image_folder_path = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

chosen_image = cv2.imread(os.path.join(image_folder_path, "ISIC_0079038.jpg"))

image = cv2.cvtColor(chosen_image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (256, 256))

kernel = np.ones((5,5), np.uint8) 

  

image = cv2.dilate(image, kernel, iterations=1) 

image = cv2.erode(image, kernel, iterations=1) 



albumentation_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),

                      A.RandomRain(p=1), A.Rotate(p=1, limit=90),

                      A.RGBShift(p=1), A.RandomSnow(p=1),

                      A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8), A.RandomContrast(limit = 0.5,p = 1),

                      A.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","GaussNoise","CLAHE",

               "RandomRain","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")

import pywt

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 



def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= circle_crop(image)

        coeffs = pywt.dwt2(image, 'bior1.3')

        IM1, (IM2, IM3, IM4) = coeffs

        image = IM2

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
import pywt

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = crop_image_from_gray(img)    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 



def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= circle_crop(image)

        coeffs = pywt.dwt2(image, 'bior1.3')

        IM1, (IM2, IM3, IM4) = coeffs

        image = IM3

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.gray) 

        axs[i,j].axis('off')

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= circle_crop(image)

        circles = cv2.HoughCircles(imagee,cv2.HOUGH_GRADIENT,1,20,

                            param1=50,param2=30,minRadius=0,maxRadius=0)



        image = np.uint16(np.around(circles))

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.gray) 

        axs[i,j].axis('off')

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15), constrained_layout=True)

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image= circle_crop(image)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,50,150,apertureSize = 3)



        lines = cv2.HoughLines(edges,1,np.pi/180,200)

        for rho,theta in lines[0]:

            a = np.cos(theta)

            b = np.sin(theta)

            x0 = a*rho

            y0 = b*rho

            x1 = int(x0 + 1000*(-b))

            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))

            y2 = int(y0 - 1000*(a))



        image = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.gray) 

        axs[i,j].axis('off')

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's growth");
X_train, X_val = train_test_split(train, test_size=0.2, random_state=42)

X_train['image_name'] = X_train['image_name'].apply(lambda x: x + '.jpg')

X_val['image_name'] = X_val['image_name'].apply(lambda x: x + '.jpg')

test['image_name'] = test['image_name'].apply(lambda x: x + '.jpg')

X_train['target'] = X_train['target'].apply(lambda x: str(x))

X_val['target'] = X_val['target'].apply(lambda x: str(x))
from keras.applications import ResNet50 as model

from PIL import Image

model = model(weights='imagenet')

model.compile(

    'Adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'],

)
print(f"Train set is {train.shape[0] / test.shape[0]} times bigger than test set.")
from sklearn.model_selection import RepeatedKFold

splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
partitions = []



for train_idx, test_idx in splitter.split(train.index.values):

    partition = {}

    partition["train"] = train.image_name.values[train_idx]

    partition["validation"] = train.image_name.values[test_idx]

    partitions.append(partition)

    print("TRAIN:", train_idx, "TEST:", test_idx)

    print("TRAIN:", len(train_idx), "TEST:", len(test_idx))
class Config:

    BATCH_SIZE = 8

    EPOCHS = 40

    WARMUP_EPOCHS = 2

    LEARNING_RATE = 1e-4

    WARMUP_LEARNING_RATE = 1e-3

    HEIGHT = 224

    WIDTH = 224

    CANAL = 3

    N_CLASSES = train['target'].nunique()

    ES_PATIENCE = 5

    RLROP_PATIENCE = 3

    DECAY_DROP = 0.5
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 

                                 rotation_range=360,

                                 horizontal_flip=True,

                                 vertical_flip=True)



train_generator=train_datagen.flow_from_dataframe(

    dataframe=X_train,

    directory='../input/siim-isic-melanoma-classification/jpeg/train/',

    x_col="image_name",

    y_col="target",

    class_mode="raw",

    batch_size=Config.BATCH_SIZE,

    target_size=(Config.HEIGHT, Config.WIDTH),

    seed=0)



validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



valid_generator=validation_datagen.flow_from_dataframe(

    dataframe=X_val,

    directory='../input/siim-isic-melanoma-classification/jpeg/train/',

    x_col="image_name",

    y_col="target",

    class_mode="raw", 

    batch_size=Config.BATCH_SIZE,   

    target_size=(Config.HEIGHT, Config.WIDTH),

    seed=0)



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(  

        dataframe=test,

        directory = '../input/siim-isic-melanoma-classification/jpeg/test/',

        x_col="image_name",

        batch_size=1,

        class_mode=None,

        shuffle=False,

        target_size=(Config.HEIGHT, Config.WIDTH),

        seed=0)
"""

%%time

STEP_SIZE_TRAIN = train_generator.n//64

STEP_SIZE_VALID = valid_generator.n//64

history = model.fit_generator(generator=train_generator,

                                    steps_per_epoch=STEP_SIZE_TRAIN,

                                    validation_data=valid_generator,

                                    validation_steps=STEP_SIZE_VALID,

                                    epochs=1,

                                    verbose=1).history

"""                                    
import random

def sp_noise(image,prob):

    '''

    Add salt and pepper noise to image

    prob: Probability of the noise

    '''

    output = np.zeros(image.shape,np.uint8)

    thres = 1 - prob 

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()

            if rdn < prob:

                output[i][j] = 0

            elif rdn > thres:

                output[i][j] = 255

            else:

                output[i][j] = image[i][j]

    return output



def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image = sp_noise(image, 0.192)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='lentigo NOS']['image_name'], title="Lentigo NOS's Erosion");
import random

def sp_noise(image,prob):

    '''

    Add salt and pepper noise to image

    prob: Probability of the noise

    '''

    output = np.zeros(image.shape,np.uint8)

    thres = 1 - prob 

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()

            if rdn < prob:

                output[i][j] = 0

            elif rdn > thres:

                output[i][j] = 255

            else:

                output[i][j] = image[i][j]

    return output



def view_images_aug(images, title = '', aug = None):

    width = 6

    height = 5

    fig, axs = plt.subplots(height, width, figsize=(15,15))

    for im in range(0, height * width):  

        data = pydicom.read_file(os.path.join(train_images_dir, list(images)[im]+ '.dcm'))

        image = data.pixel_array

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (256, 256))

        image = sp_noise(image, 0.192)

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        

    plt.suptitle(title)

view_images_aug(train[train['diagnosis']=='unknown']['image_name'], title="Lentigo NOS's Erosion");
YouTubeVideo("bQLphecl-1A", height=500, width=800)
YouTubeVideo("=-O3fLMg6qwQ", height=500, width=800)