from tqdm import tqdm 

import numpy as np 



embeddings_index = {}

EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'

f = open(EMBEDDING_FILE, encoding="utf8")

for line in tqdm(f):

    values = line.split()

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[values[0]] = coefs

f.close()
from sklearn.metrics.pairwise import linear_kernel

from nltk.corpus import stopwords 

import pandas as pd

import string, os

import json



ignorewords = ["year", "experience", "full-time", "part-time", "part", "time", "full", "university", "college", "degree", "major"]

stopwords = stopwords.words('english')

numbs  = "0123456789"



""" function to cleanup the text """

def _cleanup(text):

    text = text.lower()

    text = " ".join([c for c in text.split() if c not in stopwords])

    for c in string.punctuation:

        text = text.replace(c, " ")

    text = " ".join([c for c in text.split() if c not in stopwords])

    words = []

    for wrd in text.split():

        if len(wrd) <= 2: 

            continue

        if wrd in ignorewords:

            continue

        words.append(wrd)

    text = " ".join(words)    

    return text



""" function to clean the filename and obtain the job role title"""

def _clean(fname):

    for num in numbs: 

        fname = fname.split(num)[0].strip()

    return fname.title()



results = []

base_path = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"

for fname in os.listdir(base_path):

    if fname == "POLICE COMMANDER 2251 092917.txt":

        continue



    txt = open(base_path + fname).read()

    lines = txt.split("\n")

    start = 0

    rel_lines = []

    for i, l in enumerate(lines):

        if 'requirement' in l.lower():

            start = i

            break

    for i, l in enumerate(lines[start+1:]):

        if "substituted" in l.lower():

            break

        if l.isupper():

            break

        rel_lines.append(l)

    req1 = " ".join(rel_lines)

    req = _cleanup(req1)

    d = {'cleaned' : req, 'original' : req1, 'title' : _clean(fname)}

    results.append(d)

    

data = pd.DataFrame(results)[['title','original','cleaned']]

data.head()
""" function to generate document vector by aggregating word embeddings """

def generate_doc_vectors(s):

    words = str(s).lower().split() 

    words = [w for w in words if w.isalpha()]

    M = []

    for w in words:

        if w in embeddings_index:

            M.append(embeddings_index[w])

    v = np.array(M).sum(axis=0)

    if type(v) != np.ndarray:

        return np.zeros(300)

    return v / np.sqrt((v ** 2).sum())



req_vectors = []

for i,r in data.iterrows():

    req_vectors.append(generate_doc_vectors(r['cleaned']))
_interactions = linear_kernel(req_vectors, req_vectors)



_edges = {}

for idx, row in data.iterrows():

    similar_indices = _interactions[idx].argsort()[:-100:-1]

    similar_items = [(_interactions[idx][i], data['title'][i]) for i in similar_indices]

    _edges[row['title']] = similar_items[:20]
""" function to identify the implicit links """

def get_treedata(k, threshold, limit):

    k = k.title()



    txt = """<h3><font color="#aa42f4">Requirement Texts: </font></h3>"""

    txt += "<h3><font color='#196ed6'>" + k + "</font></h3>"

    req = data[data['title'] == k]['original'].iloc(0)[0]

    if len(req) > 350:

        req = req[:350] + " ..."

    txt += "<p><b>Requirements: </b>" + req + "</p>"



    treedata = {"name" : k, "children" : [], "color" : '#97f4e3', "size":25, "exp" : ""}

    edges = _edges[k]

    edges = [_ for _ in edges if _[1] != k]

    edges = [_ for _ in edges if _[0] >= threshold]

    ignore = ['principal', "chief", "director", "supervisor"]

    counter = 0

    for i, edge in enumerate(edges):

        if any(upper in edge[1].lower() for upper in ignore):

            continue

        d = {"name" : edge[1], "children" : [], "color" : "red", "size":15, "exp" : edge[0]}

        treedata['children'].append(d)

        counter += 1

        if counter == limit:

            break

        txt += "<h3><font color='#f93b5e'>" + edge[1] + "(Context Similarity: "+str(round(edge[0], 2))+")</font></h3>"

        req1 = data[data['title'] == edge[1]]['original'].iloc(0)[0]

        if len(req1) > 350:

            req1 = req1[:350] + " ..."

        txt += "<b>Requirements: </b>" + req1 + ""

    return treedata, txt
from IPython.core.display import display, HTML, Javascript

import IPython.display



""" function to generate required javascript and HTML for the visualization """

def _get_js(treedata, idd):

    html = """<style>  

        .node circle {

          fill: #fff;

          stroke: steelblue;

          stroke-width: 3px;

        }

        .node text { font: 12px sans-serif; }

        .node--internal text {

          text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;

        }

        .link {

          fill: none;

          stroke: #ccc;

          stroke-width: 2px;

        }

    </style>

    <svg height='340' id='"""+idd+"""' width="760"></svg>"""



    js="""require.config({

        paths: {

            d3: "https://d3js.org/d3.v4.min"

        }

    });

    require(["d3"], function(d3) {

        var treeData ="""+json.dumps(treedata)+""";



        // set the dimensions and margins of the diagram

        var margin = {top: 40, right: 90, bottom: 50, left: 90},

            width = 660 - margin.left - margin.right,

            height = 290 - margin.top - margin.bottom;



        // declares a tree layout and assigns the size

        var treemap = d3.tree()

            .size([width, height]);



        //  assigns the data to a hierarchy using parent-child relationships

        var nodes = d3.hierarchy(treeData);



        // maps the node data to the tree layout

        nodes = treemap(nodes);



        // append the svg obgect to the body of the page

        // appends a 'group' element to 'svg'

        // moves the 'group' element to the top left margin

        var svg = d3.select('#"""+idd+"""').append("svg")

              .attr("width", width + margin.left + margin.right)

              .attr("height", height + margin.top + margin.bottom),

            g = svg.append("g")

              .attr("transform",

                    "translate(" + margin.left + "," + margin.top + ")");



        // adds the links between the nodes

        var link = g.selectAll(".link")

            .data( nodes.descendants().slice(1))

          .enter().append("path")

            .attr("class", "link")

            .attr("d", function(d) {

               return "M" + d.x + "," + d.y

                 + "C" + d.x + "," + (d.y + d.parent.y) / 2

                 + " " + d.parent.x + "," +  (d.y + d.parent.y) / 2

                 + " " + d.parent.x + "," + d.parent.y;

               });



        // adds each node as a group

        var node = g.selectAll(".node")

            .data(nodes.descendants())

          .enter().append("g")

            .attr("class", function(d) { 

              return "node" + 

                (d.children ? " node--internal" : " node--leaf"); })

            .attr("transform", function(d) { 

              return "translate(" + d.x + "," + d.y + ")"; });



        // adds the circle to the node

        node.append("image")

        .attr("xlink:href", function(d) { return "https://image.flaticon.com/icons/png/512/306/306473.png" })

        .attr("x", function(d) { return -15;})

        .attr("y", function(d) { return -15;})

        .attr("height", 30)

        .attr("width", 30);



        // adds the text to the node

        node.append("text")

          .attr("dy", ".35em")

          .attr("y", function(d) { return d.children ? -20 : 20; })

          .style("text-anchor", "middle")

          .text(function(d) { return d.data.name; })

          .attr("transform", "rotate(-10)" );        

    });"""

    

    return html, js



def _implicit(title, idd, threshold=0.88, limit = 4):

    treedata, txt = get_treedata(title, threshold, limit)

    h, js = _get_js(treedata, idd)

    h = display(HTML(h))

    j = IPython.display.Javascript(js)

    IPython.display.display_javascript(j)

    display(HTML(txt))
_implicit("Senior Administrative Clerk", idd="a8", threshold = 0.75, limit = 6)
_implicit("Chief Benefits Analyst", idd="a1")
_implicit("Ems Nurse Practitioner Supervisor", idd="a2")
_implicit("HELICOPTER MECHANIC SUPERVISOR", idd="a3")
_implicit("General Automotive Supervisor", idd="a4")
_implicit("Steam Plant Maintenance Supervisor", idd="a5")
_implicit("Wastewater Treatment Electrician Supervisor", idd="a6")