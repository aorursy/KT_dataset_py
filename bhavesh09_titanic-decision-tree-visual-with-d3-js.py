import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import export_graphviz
import sklearn
import os
from IPython.core.display import display, HTML, Javascript
from string import Template
import json
import IPython.display
%config IPCompleter.greedy=True
%matplotlib inline
# Any results you write to the current directory are saved as output.
def score_in_percent (a,b):
    return (sum(a==b)*100)/len(a)
# This creates a pandas dataframe and assigns it to the train and test variables
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# store target as Y
Y_train = train["Survived"]
train.drop(["Survived"], axis=1, inplace=True)

#concat both datasets for ease of operation
num_train = len(train)
all_data = pd.concat([train, test])

# Populating null fare value with median of train set
all_data["Fare"]=all_data["Fare"].fillna(train["Fare"].median())
# Populating null age value with median of train set
#all_data["Age"]=all_data["Age"].fillna(train["Age"].median())
# Populating missing embarked with most frequent value - S
all_data["Embarked"]=all_data["Embarked"].fillna("S")
# Creating new feature as Title
all_data['Title'] = all_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# Converting sex into binary
sex_mapping = {"male": 0, "female": 1}
all_data['Sex'] = all_data['Sex'].map(sex_mapping)
guess_age=all_data.groupby(['Title','Pclass','Sex'])['Age'].agg(['mean','count']).reset_index()
guess_age.columns= ['Title','Pclass','Sex','ga_mean','ga_cnt'] 
guess_age["ga_mean"]=guess_age["ga_mean"].fillna(28)
guess_age["ga_mean"]=guess_age["ga_mean"].astype(int)
all_data=all_data.merge(guess_age, how='left')
all_data.loc[(all_data.Age.isnull()),"Age"]=all_data[(all_data.Age.isnull())].ga_mean
# Drop columns which may cause overfit, also residual columns from above dataset
all_data.drop(["Cabin","Name","Ticket","PassengerId","ga_mean","ga_cnt"], axis=1, inplace=True)
# get dummies for categorical variables
all_data = pd.get_dummies(all_data)
X_train = all_data[:num_train]
X_test = all_data[num_train:]
X_train, X_cv, y_train, y_cv = train_test_split( X_train, Y_train, test_size = 0.3, random_state = 100)
clf_tuned = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
                                max_features=None, max_leaf_nodes=None, min_samples_leaf=10,
                                min_samples_split=10, min_weight_fraction_leaf=0.0,
                                presort=False, random_state=100, splitter='random')
clf_tuned.fit(X_train, y_train)
y_pred = clf_tuned.predict(X_cv)
y_test_pred = clf_tuned.predict(X_test)
score_in_percent(y_pred,y_cv)
def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        #count_labels = zip(clf.tree_.value[node_index, 0], labels)
        #node['name'] = ', '.join(('{} of {}'.format(int(count), label)
        #                          for count, label in count_labels))
        node['type']='leaf'
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['type']='split'
        node['label'] = '{} > {}'.format(feature, threshold)
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
        
    return node

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
cols = X_train.columns
d = rules(clf_tuned, cols, None)
with open('output.json', 'w') as outfile:  
    json.dump(d, outfile,cls=MyEncoder)

j = json.dumps(d, cls=MyEncoder)
html_string = """
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
    <script type="text/javascript" src="https://d3js.org/d3.v3.min.js"></script>
    <style type="text/css">
body {
  font-family: "Helvetica Neue", Helvetica;
}
.hint {
  font-size: 12px;
  color: #999;
}
.node rect {
  cursor: pointer;
  fill: #fff;
  stroke-width: 1.5px;
}
.node text {
  font-size: 11px;
}
path.link {
  fill: none;
  stroke: #ccc;
}
    </style>
  </head>
  <body>
    <div id="body">
      <div id="footer">
        Decision Tree viewer
        <div class="hint">click to expand or collapse</div>
        <div id="menu">
          <select id="datasets"></select>
        </div>

      </div>
    </div>    
"""
js_string="""
 var m = [20, 120, 20, 120],
    w = 1280 - m[1] - m[3],
    h = 800 - m[0] - m[2],
    i = 0,
    rect_width = 80,
    rect_height = 20,
    max_link_width = 20,
    min_link_width = 1.5,
    char_to_pxl = 6,
    root;
// Add datasets dropdown
d3.select("#datasets")
    .on("change", function() {
      if (this.value !== '-') {
        d3.json(this.value + ".json", load_dataset);
      }
    })
  .selectAll("option")
    .data([
      "-",
      "output"
    ])
  .enter().append("option")
    .attr("value", String)
    .text(String);
var tree = d3.layout.tree()
    .size([h, w]);
var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.x, d.y]; });
var vis = d3.select("#body").append("svg:svg")
    .attr("width", w + m[1] + m[3])
    .attr("height", h + m[0] + m[2] + 1000)
  .append("svg:g")
    .attr("transform", "translate(" + m[3] + "," + m[0] + ")");
// global scale for link width
var link_stoke_scale = d3.scale.linear();
var color_map = d3.scale.category10();
// stroke style of link - either color or function
var stroke_callback = "#ccc";
function load_dataset(json) {
  root = json;
  root.x0 = 0;
  root.y0 = 0;
  var n_samples = root.samples;
  var n_labels = root.value.length;
  if (n_labels >= 2) {
    stroke_callback = mix_colors;
  } else if (n_labels === 1) {
    stroke_callback = mean_interpolation(root);
  }
  link_stoke_scale = d3.scale.linear()
                             .domain([0, n_samples])
                             .range([min_link_width, max_link_width]);
  function toggleAll(d) {
    if (d && d.children) {
      d.children.forEach(toggleAll);
      toggle(d);
    }
  }
  // Initialize the display to show a few nodes.
  root.children.forEach(toggleAll);
  update(root);
}
function update(source) {
  var duration = d3.event && d3.event.altKey ? 5000 : 500;
  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse();
  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 180; });
  // Update the nodes…
  var node = vis.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });
  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
      .on("click", function(d) { toggle(d); update(d); });
  nodeEnter.append("svg:rect")
      .attr("x", function(d) {
        var label = node_label(d);
        var text_len = label.length * char_to_pxl;
        var width = d3.max([rect_width, text_len])
        return -width / 2;
      })
      .attr("width", 1e-6)
      .attr("height", 1e-6)
      .attr("rx", function(d) { return d.type === "split" ? 2 : 0;})
      .attr("ry", function(d) { return d.type === "split" ? 2 : 0;})
      .style("stroke", function(d) { return d.type === "split" ? "steelblue" : "olivedrab";})
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
  nodeEnter.append("svg:text")
      .attr("dy", "12px")
      .attr("text-anchor", "middle")
      .text(node_label)
      .style("fill-opacity", 1e-6);
  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  nodeUpdate.select("rect")
      .attr("width", function(d) {
        var label = node_label(d);
        var text_len = label.length * char_to_pxl;
        var width = d3.max([rect_width, text_len])
        return width;
      })
      .attr("height", rect_height)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
  nodeUpdate.select("text")
      .style("fill-opacity", 1);
  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + source.x + "," + source.y + ")"; })
      .remove();
  nodeExit.select("rect")
      .attr("width", 1e-6)
      .attr("height", 1e-6);
  nodeExit.select("text")
      .style("fill-opacity", 1e-6);
  // Update the links
  var link = vis.selectAll("path.link")
      .data(tree.links(nodes), function(d) { return d.target.id; });
  // Enter any new links at the parent's previous position.
  link.enter().insert("svg:path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      })
      .transition()
      .duration(duration)
      .attr("d", diagonal)
      .style("stroke-width", function(d) {return link_stoke_scale(d.target.samples);})
      .style("stroke", stroke_callback);
  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal)
      .style("stroke-width", function(d) {return link_stoke_scale(d.target.samples);})
      .style("stroke", stroke_callback);
  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();
  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}
// Toggle children.
function toggle(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
}
// Node labels
function node_label(d) {
  if (d.type === "leaf") {
    // leaf
    var formatter = d3.format(".2f");
    var vals = [];
    d.value.forEach(function(v) {
        vals.push(formatter(v));
    });
    return "[" + vals.join(", ") + "]";
  } else {
    // split node
    return d.label;
  }
}
/**
 * Mixes colors according to the relative frequency of classes.
 */
function mix_colors(d) {
  var value = d.target.value;
  var sum = d3.sum(value);
  var col = d3.rgb(0, 0, 0);
  value.forEach(function(val, i) {
    var label_color = d3.rgb(color_map(i));
    var mix_coef = val / sum;
    col.r += mix_coef * label_color.r;
    col.g += mix_coef * label_color.g;
    col.b += mix_coef * label_color.b;
  });
  return col;
}
/**
 * A linear interpolator for value[0].
 *
 * Useful for link coloring in regression trees.
 */
function mean_interpolation(root) {
  var max = 1e-9,
      min = 1e9;
  function recurse(node) {
    if (node.value[0] > max) {
      max = node.value[0];
    }
    if (node.value[0] < min) {
      min = node.value[0];
    }
    if (node.children) {
      node.children.forEach(recurse);
    }
  }
  recurse(root);
  var scale = d3.scale.linear().domain([min, max])
                               .range(["#2166AC","#B2182B"]);
  function interpolator(d) {
    return scale(d.target.value[0]);
  }
  return interpolator;
}
 """
h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)
