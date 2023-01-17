import pandas as pd
import numpy as np
from IPython.core.display import display, HTML, Javascript
from string import Template
import json
import IPython.display
%config IPCompleter.greedy=True
train = pd.read_csv(r'../input/train.tsv',delimiter ='\t')
train.describe()
train.describe(include=['O'])
train['name'] = train['name'].str.lower()
train['category_name'] = train['category_name'].str.lower()
train['brand_name'] = train['brand_name'].str.lower()
train['item_description'] = train['item_description'].str.lower()
# reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
train['general_cat'], train['subcat_1'], train['subcat_2'] = \
zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.head()    
train_category_summary = train.groupby(['general_cat', 'subcat_1', 'subcat_2','name']).size().reset_index(name='counts')
train_category_summary = train_category_summary[train_category_summary['counts'] > 10]
data = {}
data["name"] = "Mercari Product Category Distribution"
data["children"] = []
# Split dataset into Continents:
for gc in train_category_summary['general_cat'].unique():
    gc_set = train_category_summary[train_category_summary["general_cat"]==gc]
    gc_dict = {}
    gc_dict["name"] = gc
    gc_dict["children"] = []
    for s1 in gc_set['subcat_1'].unique():
        s1_set = gc_set[gc_set['subcat_1']==s1][['subcat_2','name','counts']]
        s1_dict = {}
        s1_dict["name"] = s1
        s1_dict["children"] = []
        for s2 in s1_set['subcat_2'].unique():
            s2_set = s1_set[s1_set['subcat_2']==s2][['subcat_2','name','counts']]
            s2_dict = {}
            s2_dict['name'] = s2
            s2_dict["children"] = []
            for id in s2_set.values:
                id_dict = {}
                id_dict['name'] = id[1]
                id_dict['size'] = id[2]
                s2_dict["children"].append(id_dict)
            s1_dict['children'].append(s2_dict)
        gc_dict['children'].append(s1_dict)
    data["children"].append(gc_dict)
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
<svg width="800" height="800"></svg>
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

var color = d3.scaleSequential(d3.interpolatePlasma)
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
with open('output.json', 'w') as outfile:  
    json.dump(data, outfile)
h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)
