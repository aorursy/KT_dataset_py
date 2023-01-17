import json
import collections
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.tag import StanfordNERTagger
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse import CoreNLPParser
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from tqdm.notebook import tqdm

#define keywords
keywords_covid = ['sars-cov', 'sars', 'coronavirus',  'coronaviruses', 'ncov', 'covid-19', 'covid19', '2019-ncov', '2019-ncov.', 'wuhan', 'sars-cov-2']

term_data_path = ('/kaggle/input/covid19-terms/terms_covid_dataset.csv')

df_covid19 = pd.read_csv(term_data_path)
df_covid19.head()
covdict = collections.OrderedDict()
for i, row in tqdm(df_covid19.iterrows()):
    title = row['title']
    pid = row['Paper id']
    covdict[pid]=title
#paper_term_dict # paper_id + [long term + cval ]
#term_paper_dict # long term [paper id + c val]
#term_cval_dict  # long term cval 
#n_term_dict # single term long term
#n_single_term_value_dict # single term cval

paper_term_dict = collections.OrderedDict()
term_paper_dict = collections.OrderedDict()
term_cval_dict = collections.OrderedDict()
for i, row in tqdm(df_covid19.iterrows()):
    
    pid = row['Paper id']
    term = row['Extracted term']
    cvalue = row['C-value']
    if not pid in paper_term_dict:
        paper_term_dict[pid] = []
    paper_term_dict[pid].append([term,float(cvalue)])
    if not term in term_paper_dict:
        term_paper_dict[term] = []
    term_paper_dict[term].append([pid,float(cvalue)])
    if not term in term_cval_dict:
        term_cval_dict[term]=0.0
    term_cval_dict[term]+=float(cvalue)
stop_words = set(stopwords.words('english')) 
 
n_term_dict = collections.OrderedDict()
n_long_term_dict = collections.OrderedDict()
lemmatizer = WordNetLemmatizer()
count = 0

for term  in tqdm(term_cval_dict):
    cval_float = term_cval_dict[term]
    terms = word_tokenize(term)
    
    if cval_float > 1.0:
        for t in terms:
            if (not t.lower() in stop_words) and (len(t)>2):
                #print(t)
                t_lemma = ''
                if not t.isupper():
                    t_lemma = lemmatizer.lemmatize(t.lower())
                else:
                    t_lemma = t
                if not t_lemma in n_term_dict:
                    n_term_dict[t_lemma] = []
                n_term_dict[t_lemma].append(term)
        
    count+=1
n_single_term_value_dict = collections.OrderedDict()
for term in tqdm(n_term_dict):
    c_value = 0.0
    for long_term in n_term_dict[term]:
        c_val = term_cval_dict[long_term]
        #print(c_val)
        c_value += float(c_val)
    n_single_term_value_dict[term]= c_value
a = input("Please provide a (single word) search term:")

q_term = a
q_terms = []
if not q_term in n_term_dict:
    print("No terms found with this word")
else:
    q_terms = n_term_dict[a]
    
print(str(len(q_terms))+" terms found!")
print("Run the 3 cells below to visualise them")
###TREE###
import json

root2 = {}
root2['name']= q_term
root2['color']="#ffae00"
root2['percent']=""
root2["children"] = []
root2['value'] = ""
single_cval = n_single_term_value_dict[q_term]
root2['size']= 25


count = 0

q_terms_cval = dict((k,term_cval_dict[k]) for k in q_terms if k in term_cval_dict)
sorted_terms_cval  = collections.OrderedDict(sorted(q_terms_cval.items(), reverse=True, key=lambda t: t[1]))

for t in sorted_terms_cval:
    
    tc_val = int(float(term_cval_dict[t]))
    #print(tc_val)
    
    new_child = {
        'name' : t + " : " + str(tc_val),
        "color": "#f08080",
        "percent": str(tc_val/single_cval),
        "children": [],
        "value" : tc_val,
        "size" : 25}
    
    
    scount = 0
    paper_cvals = term_paper_dict[t] 
    sorted_paper_cvals = s = sorted(paper_cvals, reverse=True, key = lambda x: (x[1]))#sorted(paper_cvals, key = itemgetter(1))
    for paper_cval in sorted_paper_cvals:
        paper = paper_cval[0]
        #print(paper)
        cval = float(paper_cval[1])
        
        if cval>1.0 and scount<10:
            title = paper
            if paper in covdict:
                title = covdict[paper]
                #print(title)
            if len(str(title))>1:
                new_child['children'].append({
                    'name':str(int(cval))+" : "+ str(title) + "   (paper id: "+paper+")",
                    "color": "#239b56",
                    "percent": str(cval/tc_val),
                    "children":[],
                    "value": str(cval),
                    "size":20
                })
                scount+=1
        
    if count<20:
        root2['children'].append(new_child)
        count+=1

with open('tree.json', 'w') as outfile:
    json.dump(root2, outfile)
html_d1 = """<!DOCTYPE html><style>.node text {font: 14px sans-serif;}.link {fill: none;stroke: #ccc;stroke-width: 2px;}</style><svg id="four" width="1200" height="900" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""
js_d1="""
require(["d3"], function(d3) {
var treeData = """ +json.dumps(root2) + """
var root, margin = {
        top: 20,
        right: 10,
        bottom: 120,
        left: 120
    },
    width = 1500 - margin.left - margin.right,
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

h = display(HTML(html_d1))
j = IPython.display.Javascript(js_d1)
IPython.display.display_javascript(j)
import random


paper_id = input("Please provide a paper id (or a random will be chosen):")
if not paper_id in covdict:
    print("Not found: choosing a random article")
    paper_id = random.choice(list(covdict.keys()))
print("The selected article title is: "+covdict[paper_id] + "with paper id : " + paper_id)
print("Run the cells below to visualise the most important terms for this article")


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.core.display import display, HTML, Javascript
from string import Template
import pandas as pd
import numpy as np
import json, random
import IPython.display
import warnings
warnings.filterwarnings('ignore')
terms_val = paper_term_dict[paper_id]
terms = []
term_str = "id,value\n"
for termval in terms_val:
    term = termval[0]
    val = termval[1]
    if not term in terms:
        terms.append(term)
        if val>2.0:
            term_str+=term+","+str(round(float(val)))+"\n"
fout = open("texts.csv", "w")
fout.write(term_str)
fout.close()

html_p1 = """<!DOCTYPE html><svg id="one" width="1200" height="900" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""
js_p1 = """require.config({paths: {d3: "https://d3js.org/d3.v4.min"}});
require(["d3"], function(d3) {var svg=d3.select("#one"),width=+svg.attr("width"),height=+svg.attr("height"),format=d3.format(",d"),color=d3.scaleOrdinal(d3.schemeCategory20c);console.log(color);var pack=d3.pack().size([width,height]).padding(1.5);
d3.csv("texts.csv",function(t){if(t.value=+t.value,t.value)return t},function(t,e){if(t)throw t;var n=d3.hierarchy({children:e}).sum(function(t){return t.value}).each(function(t){if(e=t.data.id){var e,n=e.lastIndexOf(".");t.id=e,t.package=e.slice(0,n),t.class=e.slice(n+1)}}),a=(d3.select("body").append("div").style("position","absolute").style("z-index","10").style("visibility","hidden").text("a"),svg.selectAll(".node").data(pack(n).leaves()).enter().append("g").attr("class","node").attr("transform",function(t){return"translate("+t.x+","+t.y+")"}));a.append("circle").attr("id",function(t){return t.id}).attr("r",function(t){return t.r}).style("fill",function(t){return color(t.package)}),a.append("clipPath").attr("id",function(t){return"clip-"+t.id}).append("use").attr("xlink:href",function(t){return"#"+t.id}),a.append("svg:title").text(function(t){return t.value}),a.append("text").attr("clip-path",function(t){return"url(#clip-"+t.id+")"}).selectAll("tspan").data(function(t){return t.class.split(/(?=[^a-z][^A-Z])/g)}).enter().append("tspan").attr("x",0).attr("y",function(t,e,n){return 13+10*(e-n.length/2-.5)}).text(function(t){return t})});});
"""
h = display(HTML(html_p1))
j = IPython.display.Javascript(js_p1)
IPython.display.display_javascript(j)
