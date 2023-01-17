# TODO: Fix this up.
import pandas as pd

pd.set_option('max_columns', None)



casting = pd.read_csv("../input/AllMoviesCastingRaw.csv", encoding='utf-8-sig', sep=";")

import numpy as np

casting.loc[:, ['actor1_name', 'actor2_name', 'actor3_name', 

                'actor4_name', 'actor5_name']

           ].replace('none', np.nan).head()



cary_grant = casting.query('actor1_name == "Cary Grant" | actor2_name == "Cary Grant"'

              '| actor3_name == "Cary Grant" | actor4_name == "Cary Grant"'

              '| actor5_name == "Cary Grant"')



import itertools

actor_pairs = []



for _, srs in cary_grant.loc[:, 

                          ['actor1_name', 'actor2_name', 'actor3_name',

                           'actor4_name', 'actor5_name']

                         ].iteritems():

    actor_pairs.append(list(

            itertools.combinations(srs.replace('none', np.nan).dropna().values, 2)

        )

    )



actor_pairs = set([tuple(set(v)) for v in actor_pairs])

actor_pairs = list(itertools.chain(*actor_pairs))

actor_pairs = pd.DataFrame(np.asarray(actor_pairs))

actor_pairs = actor_pairs.rename(columns={0: 'actor_one', 1: 'actor_two'})



import networkx as nx



G = nx.Graph()

G.add_node('Cary Grant')



def apg(name):

    G.add_node(name)

    G.add_edge('Cary Grant', name)



actor_pairs.query('actor_one == "Cary Grant" | actor_two == "Cary Grant"').apply(

    lambda srs: apg(srs['actor_one']) if srs['actor_two'] == 'Cary Grant' else apg(srs['actor_two']),

    axis='columns'

)

pass



print("Cary Grant co-starred with {0} different movie stars in this dataset.".format(

    len(G.edges()))

     )



from networkx.readwrite import json_graph

cary_grant_json = json_graph.node_link_data(G)



import json



cary_grant_json['nodes'] = list(map(

    lambda n: {'id': n['id'], 'group': 2}, cary_grant_json['nodes']

))

cary_grant_json['links'] = list(map(

    lambda st: {'source': cary_grant_json['nodes'][st['source']]['id'], 

                'target': cary_grant_json['nodes'][st['target']]['id'],

                'value': 2},

    cary_grant_json['links'])

    )



import json

with open("cary_grant.json", "w") as f:

    f.write(json.dumps(cary_grant_json))
html_string = """

<!DOCTYPE html>

<meta charset="utf-8">

<style>



.links line {

  stroke: #999;

  stroke-opacity: 0.6;

}



.nodes circle {

  stroke: #fff;

  stroke-width: 1.5px;

}



</style>

<svg width="800" height="800"></svg>

"""
js_string = """

require.config({

    paths: {

        d3: "https://d3js.org/d3.v4.min"

     }

 });



require(["d3"], function(d3) {

  

  console.log(d3);



var svg = d3.select("svg"),

    width = +svg.attr("width"),

    height = +svg.attr("height");



var color = d3.scaleOrdinal(d3.schemeCategory20);



var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }))

    .force("charge", d3.forceManyBody())

    .force("center", d3.forceCenter(width / 2, height / 2));



d3.json("cary_grant.json", function(error, graph) {

  if (error) throw error;



  var link = svg.append("g")

      .attr("class", "links")

    .selectAll("line")

    .data(graph.links)

    .enter().append("line")

      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });



  var node = svg.append("g")

      .attr("class", "nodes")

    .selectAll("circle")

    .data(graph.nodes)

    .enter().append("circle")

      .attr("r", 5)

      .attr("fill", function(d) { return color(d.group); })

      .call(d3.drag()

          .on("start", dragstarted)

          .on("drag", dragged)

          .on("end", dragended))



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
from IPython.core.display import display, HTML, Javascript

import IPython.display



h = display(HTML(html_string))

j = IPython.display.Javascript(js_string)

IPython.display.display_javascript(j)