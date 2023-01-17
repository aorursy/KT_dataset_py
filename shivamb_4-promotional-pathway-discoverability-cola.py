from IPython.core.display import display, HTML, Javascript

import IPython.display

import json, os

import random 



""" define global variables, maps, and lookup dictionaries to be used in the program """

colors = ['red', 'blue','green', 'yellow', 'pink', 'gray', 'orange', 'purple', 'brown', 'silver','mistygreen','chocolate', 'peru', 'indigo', 'deeppink', 'linen', 'lavender', 'snow', 'indianred', 'lawngreen', 'papayawhip', 'mediumturquoise', 'plum', 'tan', 'magenta', 'wheat', 'aqua', 'cadetblue', 'forestgreen', 'honeydew', 'ivory', 'orchid']

colors = ["#80f2e6", "#84f984", "#9e86ef", "#f9b083", "#ed82ab", "#ff5661", "#82cef2", "#ef91f2", "#fcf344", "#b59f98", "#727171", "#a87000", "#a2a800", "#7ba800", "#00a80b", "#1d78c6", "#535e77", "#dbd9f9", "#58286d", "#b100ff", "#ff2dd1", "#dbb1d"]

n_map  = {"one" : "1", "two" : "2", "three" : "3" , "four" : "4", 'five' : "5", "six" : "6", "seven" : "7"}

levels = ['X', 'IX', 'VIII', 'VII', 'VI', 'V', 'IV', 'III', 'II', "I"]

path   = "../input/cityofla/CityofLA/Job Bulletins/"

flags  = ['as a', 'the level of', ' as ']

numbs  = "0123456789"

files  = os.listdir(path)



random.shuffle(colors)



""" function to clean the filename and obtain the job role title"""

def _clean(fname):

    for num in numbs: 

        fname = fname.split(num)[0].strip()

    return fname.lower()



""" function to clean the position-level from a job role title """

def gclean(x):

    for level in levels:

        x = x.replace(" "+level+" ","").strip()

    return x



""" get the value of year from the experience related requirement """

def getyear(txt):

    year = ""

    if "university" in txt:

        txt = txt.split("university")[1]

    txt = txt.split("year")[0]

    if "." in txt:

        txt = txt.split(".")[1]

    elif "and " in txt:

        txt = txt.split("and ")[1]

    if txt.strip().lower() in n_map:

        year = n_map[txt.strip().lower()]

    return year

    

""" function to search and extract the job roles mentioned in the requirement line related to experience """

def _get_roles_required(fname):

    

    ## extract the requirement portion

    txt = open(path + fname).read()

    lines = txt.split("\n")

    for i, x in enumerate(lines):

        if "requirement" in x.lower():

            start = i

            break

    for i, x in enumerate(lines[start+1:]):

        if x.isupper():

            end = i

            break

    

    ## check if any role is mentioned

    results = {}

    years = []

    for i, x in enumerate(lines[start+1:start+end]):

        if not any(_ in x for _ in flags):

            continue

        for r in roles:

            for level in levels:

                if r.title() + " " + level in x:

                    relt = x.split(r.title()+" "+level)[0]

                    results[r.title() + " " + level] = getyear(relt)

                elif r.title() in x:

                    relt = x.split(r.title())[0]

                    results[r.title()] = getyear(relt)

    

    ## remove redundant roles 

    removals = []

    keys = list(results.keys())

    for l1 in keys:

        for l2 in keys:

            if l1 == l2:

                continue

            if l1 in l2:

                removals.append(l1)

    for rem in list(set(removals)):

        del results[rem]

    return results



## create a global mapping of all the roles and their required job roles

roles = [_clean(f) for f in files]

roles = [x for x in roles if x]



doc = {}

for f in files:

    if _clean(f) == "":

        continue

    try:

        res = _get_roles_required(f)

        doc[_clean(f).title()] = res

    except Exception as E:

        pass

    

""" function to return node size based on experience """  

def map_size(num):

    if num == "":

        return 10

    num = int(num)

    doc = {1:10, 2:15, 3:23, 4:32, 5:39, 6:28}    

    if num in doc:

        return doc[num]

    else:

        return 40

    

""" function to generate the relevant data for the tree like visualization """

def get_tree_data(key, ddoc):

    visited = {}

    cnt = 0

    v = ddoc[key]

    ## level 0

    

    treedata = {"name" : key, "children" : [], "color" : '#97f4e3', "size":40, "exp" : ""}

    for each in v:

        exp = str(v[each]) + "y"

        size = map_size(v[each])

        

        if key == each:

            continue

        

        ## level 1

        c_each = gclean(each)

        if c_each not in ddoc:

            if c_each not in visited:

                visited[each] = colors[cnt]

                cnt += 1

            d = {"name" : each, "children" : [], "color" : visited[each], "size":size, "exp" : exp}

            treedata['children'].append(d)

        

        if c_each in ddoc:

            if each not in visited:

                visited[each] = colors[cnt]

                cnt += 1

            

            d = {"name" : each, "children" : [], "color" : visited[each], "size":size, "exp" : exp}

            for each1 in ddoc[c_each]:

                if each1 not in visited:

                    visited[each1] = colors[cnt]

                    cnt += 1

                

                exp = str(ddoc[c_each][each1]) + "y"

                size = map_size(ddoc[c_each][each1])

                m = {"name" : each1, "children" : [], "color" : visited[each1], "size":size, "exp" : exp}

                

                ## level 2 

                c_k = gclean(each1)

                if c_k in ddoc:

                    for each2 in ddoc[c_k]:

                        if each2 not in visited:

                            visited[each2] = colors[cnt]

                            cnt += 1

                        

                        exp = str(ddoc[c_k][each2]) + "y"

                        size = map_size(ddoc[c_k][each2])

                        p = {'name' : each2, "children" : [], "color" : visited[each2], "size":size, "exp" : exp}

                        m['children'].append(p)

                d['children'].append(m)

            treedata['children'].append(d)

    return treedata



""" function to generate required javascript and HTML for the visualization """

def _get_js(treedata, div_id, rot, small = False):

    rt = ""

    if rot == True:

        rt = """.attr("transform", "rotate(-10)" )"""

    ht = 610

    if small:

        ht = 360

    

    html = """<style>  

        .node circle {

         stroke-width: 4px;

        }

        .node text { font: 11px sans-serif; }

        .node--internal text {

         text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;

        }

        .link {

         fill: none;

         stroke: #ccc;

        }

    </style>

    <svg height='"""+str(ht)+"""' id='"""+div_id+"""' width="760"></svg>"""



    js="""require.config({

        paths: {

            d3: "https://d3js.org/d3.v4.min"

        }

    });

    require(["d3"], function(d3) {

        var treeData = """ +json.dumps(treedata)+ """;



        var margin = {top: 40, right: 30, bottom: 50, left: 90},

            width = 760 - margin.left - margin.right,

            height = '"""+str(ht-50)+"""' - margin.top - margin.bottom;



        var treemap = d3.tree().size([width, height]);

        var nodes = d3.hierarchy(treeData);

        nodes = treemap(nodes);

        var svg = d3.select('#"""+div_id+"""').append("svg").attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom),

            g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var link = g.selectAll(".link").data(nodes.descendants().slice(1)).enter().append("path").attr("class", "link").attr("d", function(d) {

            return "M" + d.x + "," + d.y + "C" + d.x + "," + (d.y + d.parent.y) / 2 + " " + d.parent.x + "," + (d.y + d.parent.y) / 2 + " " + d.parent.x + "," + d.parent.y;

        }).attr("stroke-width", function(d) {

            return (7);

        });

        var node = g.selectAll(".node").data(nodes.descendants()).enter().append("g").attr("class", function(d) {

            return "node" + (d.children ? " node--internal" : " node--leaf");

        }).attr("transform", function(d) {

            return "translate(" + d.x + "," + d.y + ")";

        });

        node.append("circle").attr("r", function(d){ return d.data.size }).style("fill", function(d) {

            return d.data.color;

        });

        

        node.append("text")

              .attr("text-anchor", "middle")

              .attr("dy", ".35em")

              .text(function(d) { return d.data.exp; });

        node.append("text").attr("dy", ".15em").attr("y", function(d) {

            return d.children ? -20 : 20;

        }).style("text-anchor", "middle").text(function(d) {

            var name = d.data.name;

            return name;

        })"""+rt+""";

        

    });"""

    return html, js



def getjob(key, idd, rot, small = False):

    treedata = get_tree_data(key, doc)

    html, js = _get_js(treedata, idd, rot, small)

    h = display(HTML(html))

    j = IPython.display.Javascript(js)

    IPython.display.display_javascript(j)
getjob("Senior Systems Analyst", "id1", rot=False)
getjob("Chief Of Airport Planning", "id2", rot=False)
getjob("Water Utility Superintendent", "id3", rot=True)
getjob("Chief Inspector", "id4", rot=True)
getjob("Senior Equipment Mechanic", "id11", rot=False, small = True)

getjob("Housing Inspector", "id12", rot=False, small = True)
getjob("Senior Housing Inspector",      "plot_id2", rot=True, small = True)

getjob("Management Analyst",            "plot_id3", rot=True, small = True)

getjob("Director Of Printing Services", "plot_id4", rot=True, small = False)
rdoc = {}

for k,v in doc.items():

    for each in v:

        if each not in rdoc:

            rdoc[each] = []

        if each != k:

            rdoc[each].append(k)

            

""" function to generate the relevant data for the tree like visualization """

def get_tree_data_promotion(key, ddoc):

    visited = {}

    cnt = 0

    v = ddoc[key]

    ## level 0

    

    treedata = {"name" : key, "children" : [], "color" : '#97f4e3', "size":40, "exp" : ""}

    for each in v:

        if key == each:

            continue

        

        ## level 1

        c_each = gclean(each)

        if c_each not in ddoc:

            if c_each not in visited:

                visited[each] = colors[cnt]

                cnt += 1

            d = {"name" : each, "children" : [], "color" : visited[each]}

            treedata['children'].append(d)

        

        if c_each in ddoc:

            if each not in visited:

                visited[each] = colors[cnt]

                cnt += 1

            

            d = {"name" : each, "children" : [], "color" : visited[each]}

            for each1 in ddoc[c_each]:

                if each1 not in visited:

                    visited[each1] = colors[cnt]

                    cnt += 1

                

                m = {"name" : each1, "children" : [], "color" : visited[each1]}

                

                ## level 2 

                c_k = gclean(each1)

                if c_k in ddoc:

                    for each2 in ddoc[c_k]:

                        if each2 not in visited:

                            visited[each2] = colors[cnt]

                            cnt += 1

                        

                        p = {'name' : each2, "children" : [], "color" : visited[each2]}

                        m['children'].append(p)

                d['children'].append(m)

            treedata['children'].append(d)

    return treedata





def _get_js2(treedata, div_id, rot):

    rt = ""

    if rot == True:

        rt = """.attr("transform", "rotate(-20)" )"""

    

    html = """

    <svg height="510" id='"""+div_id+"""' width="860"></svg>"""



    js="""require.config({

        paths: {

            d3: "https://d3js.org/d3.v4.min"

        }

    });

    require(["d3"], function(d3) {

        var treeData = """ +json.dumps(treedata)+ """;

        

    // set the dimensions and margins of the diagram

    var margin = {top: 20, right: 130, bottom: 30, left: 120},

        width = 660 - margin.left - margin.right,

        height = 500 - margin.top - margin.bottom;



    // declares a tree layout and assigns the size

    var treemap = d3.tree()

        .size([height, width]);



    //  assigns the data to a hierarchy using parent-child relationships

    var nodes = d3.hierarchy(treeData, function(d) {

        return d.children;

      });



    // maps the node data to the tree layout

    nodes = treemap(nodes);



    var svg = d3.select('#"""+div_id+"""').append("svg")

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

           return "M" + d.y + "," + d.x

             + "C" + (d.y + d.parent.y) / 2 + "," + d.x

             + " " + (d.y + d.parent.y) / 2 + "," + d.parent.x

             + " " + d.parent.y + "," + d.parent.x;

           }).attr("stroke-width", function(d) {

            return (7);

        });



    // adds each node as a group

    var node = g.selectAll(".node")

        .data(nodes.descendants())

      .enter().append("g")

        .attr("class", function(d) { 

          return "node" + 

            (d.children ? " node--internal" : " node--leaf"); })

        .attr("transform", function(d) { 

          return "translate(" + d.y + "," + d.x + ")"; });



    // adds the circle to the node

    node.append("circle")

      .attr("r", function(d) { return 12; })

      .style("stroke", function(d) { return 2; })

      .style("fill", function(d) { return d.data.color; });



    // adds the text to the node

    node.append("text")

      .attr("dy", ".35em")

      .attr("x", function(d) { return d.children ? 

        (d.data.value + 4) * -1 : d.data.value + 4 })

      .style("text-anchor", function(d) { 

        return d.children ? "end" : "start"; })

      .text(function(d) { return d.data.name; })"""+rt+""";;



     });"""



    return html, js



def getjob_junior(key, idd, rot):

    treedata = get_tree_data_promotion(key, rdoc)

    html, js = _get_js2(treedata, idd, rot)

    h = display(HTML(html))

    j = IPython.display.Javascript(js)

    IPython.display.display_javascript(j)

getjob_junior("City Planner".title(), "id7", rot=True)
getjob_junior("Police Officer".title(), "id9", rot=True)
getjob_junior("Electrical Mechanic".title(), "id10", rot=True)
getjob_junior("Secretary", "id5", rot=True)