!conda install -y -c conda-forge graph-tool matplotlib
# This line fixes an issue with graph_tool installation on the current Kaggle kernel

!apt-get install libsigc++-2.0-0v5
!conda install -y -c conda-forge ipython jupyter numpy
!conda install -y -c conda-forge rdflib
# Import all necessary modules

import rdflib

import graph_tool.all as gt

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Import the graph data

# This takes a while...

g = rdflib.Graph()

g.parse('/kaggle/input/covid19-literature-knowledge-graph/kg.nt', format='nt')
# In this kernel the analysis results are interpreted from the perspective of citation networks

# Therefore, it is important to reduce the data only to citation networks

for p1, _, p2 in g.triples((None, rdflib.URIRef("http://purl.org/spar/cito/isCitedBy"), None)):

    tr = (p2, rdflib.URIRef("http://purl.org/spar/cito/cites"), p1)

    if tr not in g:

        g.add(tr)

    g.remove((p1, rdflib.URIRef("http://purl.org/spar/cito/isCitedBy"), p2))
import urllib

import pandas as pd

from tqdm.notebook import tqdm as ntqdm

metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

dois = metadata['doi'].dropna().apply(lambda x: 'http://dx.doi.org/' + x.strip('doi.org').strip('http://dx.doi.org/')).values

dois = list(set(dois))



papers = []

for doi in ntqdm(dois):

    if len(list(g.triples((rdflib.URIRef(doi), rdflib.URIRef("http://purl.org/spar/cito/cites"), None)))) > 0:

        papers.append(doi)

print(len(papers))
# OPTIONAL: Display all the predicates to find out what kind of links we can analyse between the nodes

#for pr in set(g.predicates()):

#    print(pr)
# OPTIONAL: For example, here we can see a list of all the publishers that are included in the dataset

#all_publishers = []

#for s, p, o in g.triples((None, rdflib.URIRef("https://www.ica.org/standards/RiC/ontology#publishedBy"), None)):

#    if o not in all_publishers:

#        all_publishers.append(o)

#        print(o)
# OPTIONAL:

# Print a couple of papers that we could use as root to generate the network below

#glist=list(g.triples((None, rdflib.URIRef("http://purl.org/spar/cito/cites"), None)))[:5]

#for p in glist:

#    print(p)
# This function converts the imported graph data into a graph_tool network

def create_sub_graph_gt(root, depth):

    it = 0

    vnames_inv = {"??INIT??": -1}

    # The vnames array is important! It is used to get the node label (which is a string) 

    # from the node id (which is an integer).

    vnames = {"-1": "??INIT??"}

    

    objects = set()

    

    gt_graph = gt.Graph()

    gt_graph.set_directed(True)

    

    to_explore = {root}

    for _ in range(depth):

        new_explore = set()

        for node in to_explore:

            for s, p, o in g.triples((node, rdflib.URIRef("http://purl.org/spar/cito/cites"), None)):



                s_name=str(s)

                o_name=str(o)

                

                if s_name != o_name:

                

                    if s_name not in vnames_inv:

                        vnames_inv[s_name] = it

                        vnames[str(it)] = s_name

                        gt_graph.add_vertex()

                        it=it+1

                    if o_name not in vnames_inv:

                        vnames_inv[o_name] = it

                        vnames[str(it)] = o_name

                        gt_graph.add_vertex()

                        it=it+1

                

                    v1 = gt_graph.vertex( vnames_inv[s_name] )

                    v2 = gt_graph.vertex( vnames_inv[o_name] )

                    gt_graph.edge(v1,v2,add_missing=True)

                

                new_explore.add(o)

        to_explore = new_explore

    return gt_graph, vnames, vnames_inv

 

# Get all the triples that are maximally n_h hops away from our randomly picked paper rand_paper 

rand_paper = rdflib.URIRef('http://dx.doi.org/10.1186/s12879-015-1251-y')

n_h = 100

# Generate the network based on the rdflib.Graph()

gt_graph, vnames, vnames_inv = np.array(create_sub_graph_gt(rand_paper, n_h))

# If, by any chance, we have parallel edges (i.e. a paper citing the same paper twice) 

# or self-loops (i.e. a paper citing itself), remove them

gt.remove_parallel_edges(gt_graph)

gt.remove_self_loops(gt_graph)

# Create an array of the node ids (or vertex ids) of the network

v_array=gt_graph.get_vertices()

print(v_array)
# Scatter plot    

def plot_scatter(x, y, ylabel, xlabel="node id", scale=None, inset=None, figsz=None, xlim=None, ylim=None):

    if not figsz:

        figsz = (8, 5.5)

    fig = plt.figure(figsize=figsz)

    

    plt.xticks(fontsize=14)  

    plt.yticks(fontsize=14) 

    

    plt.xlabel(xlabel, fontsize=16)  

    plt.ylabel(ylabel, fontsize=16)

    

    ax = plt.gca()  

    ax.get_xaxis().tick_bottom()  

    ax.get_yaxis().tick_left() 

      

    if scale == "log":

        ax.set_xscale('log')

        ax.set_yscale('log') 

        ax.set_xlim([min(x)+1,max(x)+1])

        ax.set_ylim([min(y)+1,max(y)+1])

        

    if xlim:

        plt.xlim(xlim)

    if ylim:

        plt.ylim(ylim)

    

    plt.scatter(x, y, color="#004080", edgecolors='black', alpha=0.75) 

        

    plt.show()

    

#Histogram

def plot_hist(data, xlabel, ylabel="node count", bin_count=None, scale=None, inset=False, figsz=None, xlim=None):

    if not figsz:

        figsz = (8, 5.5)

    plt.figure(figsize=figsz)

    

    ax = plt.subplot(111)  

    ax.get_xaxis().tick_bottom()  

    ax.get_yaxis().tick_left()  

    

    plt.xticks(fontsize=14)  

    plt.yticks(fontsize=14) 

    

    plt.xlabel(xlabel, fontsize=16)  

    plt.ylabel(ylabel, fontsize=16)

      

    if scale == "log":

        plt.xscale("log")

        plt.yscale("log")

        if not bin_count:

            bin_count = 2 * int( len(tot_degs_arr)**.5  )

    elif not bin_count:

        bin_count = int( ( max(tot_degs_arr) - min(tot_degs_arr) ) / 10 )

        

    if inset:

        inset = plt.axes([.35, .3, .5, .5])

        inset.get_xaxis().tick_bottom()  

        inset.get_yaxis().tick_left()  

        inset.set_xscale("log")

        inset.set_yscale("log")

        bin_count = 2 * int( len(tot_degs_arr)**.5  )

        inset.hist(data, color="#004080", edgecolor='black', alpha=0.75, bins=bin_count)

        

    if xlim:

        plt.xlim(xlim)

    

    ax.hist(data, color="#004080", edgecolor='black', alpha=0.75, bins=bin_count) 

        

    plt.show()

    

# Network

def plot_graph(g, pos, vmap=None):

    if not vmap:

        vmap = g.degree_property_map("total")

        vmap.a = 4 * ( np.sqrt(vmap.a) * 0.5 + 0.4)

    gt.graph_draw(g, pos=pos, vertex_fill_color='#004080', vertex_size=vmap, \

                  vertex_halo=True, vertex_halo_color='black', vertex_halo_size=1.1, \

                  edge_color='gray')
plot_graph(gt_graph, gt.arf_layout(gt_graph, max_iter=100, dt=1e-4))
# Another good-looking visualization

plot_graph(gt_graph, gt.sfdp_layout(gt_graph))
# This is how we can get different info about a paper from vnames, 

# provided the data set contains that info

node_id = 0

# DOI

paper_doi = vnames[str(node_id)]

print("DOI: ", paper_doi)



# Paper reference in the knowledge graph based on the DOI

paper_ref = rdflib.URIRef(paper_doi)



# Authors

pred_firstname = rdflib.term.URIRef('http://xmlns.com/foaf/0.1/firstName')

paper_author_fn = list(g.triples((paper_ref, pred_firstname, None)))

if paper_author_fn:

    paper_author_fn = paper_author_fn[0][2]

    

pred_surname = rdflib.term.URIRef('http://xmlns.com/foaf/0.1/surname')

paper_author_sn = list(g.triples((paper_ref, pred_surname, None)))

if paper_author_sn:

    paper_author_sn = paper_author_sn[0][2]



if paper_author_fn or paper_author_sn:

    print("Author(s): %s, %s" % (paper_author_fn, paper_author_sn))

    

# Creator

pred_creator = rdflib.term.URIRef('http://purl.org/spar/pro/creator')

paper_creator = list(g.triples((paper_ref, pred_creator, None)))

if paper_creator:

    paper_creator = paper_creator[0][2]

    print("Creator: ", paper_creator)

    

# Publisher

pred_publisher = rdflib.term.URIRef('https://www.ica.org/standards/RiC/ontology#publishedBy')

paper_publisher = list(g.triples((paper_ref, pred_publisher, None)))

if paper_publisher:

    paper_publisher = paper_publisher[0][2]

    print("Publisher: ", paper_publisher)

    

# Title

# Here we define a function that we will use later in this notebook

def get_title(paper_ref):

    pred_title = rdflib.term.URIRef('http://purl.org/dc/terms/title')

    paper_title = list(g.triples((paper_ref, pred_title, None)))

    if paper_title:

        paper_title = paper_title[0][2]

        return paper_title

    return None



print("Paper title: '%s'" % get_title(paper_ref))

# Let's first have a look at the in-degree distribution

in_degs_arr = gt_graph.get_in_degrees(gt_graph.get_vertices())

# This shows us the node with the highest in-degree

highest_in_deg_v = np.where(in_degs_arr == in_degs_arr.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(highest_in_deg_v)]) 

print("Article with top prestige: '%s'" % (get_title(paper_ref)))

# Plot the distribution

plot_scatter(v_array, in_degs_arr, 'in-degree')
# Now, let's do the same for the out-degree

out_degs_arr = gt_graph.get_out_degrees(gt_graph.get_vertices())

highest_out_deg_v = np.where(out_degs_arr == out_degs_arr.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(highest_out_deg_v)]) 

print("Article citing the most: '%s'" %  (get_title(paper_ref)))

# Plot the distribution

plot_scatter(v_array, out_degs_arr, 'out-degree')
tot_degs_arr = gt_graph.get_total_degrees(gt_graph.get_vertices())

high_deg_v=np.where(tot_degs_arr == tot_degs_arr.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_deg_v)]) 

print("Article with the highest number of links: '%s'" % (get_title(paper_ref)))

# Plot the distribution 

# including an inset that shows the same distribution but on a log-log scale

plot_hist(tot_degs_arr, "total degree", inset=True)
# An alternative way of plotting the histogram in log-log scale

total_hist = gt.vertex_hist(gt_graph, "total")

plot_scatter(total_hist[1][:-1], total_hist[0], "node count", \

             xlabel="total degree", scale="log", xlim=[0.8, max(total_hist[1][:-1])+1e+2], ylim=[0.8, max(total_hist[0])+1e+3])
# The node with the highest degree is chosen as root

root_vertex=gt_graph.vertex(high_deg_v)

plot_graph(gt_graph, gt.radial_tree_layout(gt_graph, root_vertex))
close_map_v=gt.closeness(gt_graph)

close_array_v=close_map_v.a

# The following line replaces 'nan' values with '0'

close_array_v = np.nan_to_num(close_array_v)

# Plot the distribution

plot_scatter(v_array, close_array_v, 'closeness')

# Which one is among the highest

high_close_v=np.where(close_array_v == close_array_v.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_close_v)]) 

print("Article with the highest closeness centrality is: '%s'" % (get_title(paper_ref)))

if not (not high_deg_v) and high_close_v==high_deg_v:

    print("Note that this is the same paper as the one with the highest prestige as determined by the degree centrality above.")
close_array_v_sorted=np.sort(close_array_v)

# Plotting in ascending order, we can see that the fast majority of nodes has closeness centrality values close to zero

plot_scatter(v_array, close_array_v_sorted, 'closeness', xlabel='ids of sorted array')

# Uncomment the line below to zoom into the last 200 data points and see that the closeness increases partially super-linearly

#plot_scatter(v_array, close_array_v_sorted, 'closeness', xlabel='ids of sorted array', xlim=[len(close_array_v_sorted)-200,len(close_array_v_sorted)])
betw_map_v, betw_map_e=gt.betweenness(gt_graph)

betw_array_v=betw_map_v.a



plot_scatter(v_array, betw_array_v, 'betweenness', ylim=[min(betw_array_v)-1e-4,max(betw_array_v)+1e-4])



# Get one among the highest

high_betw_v=np.where(betw_array_v == betw_array_v.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_betw_v)]) 

print("Article with the highest betweenness centrality is: '%s'" % (get_title(paper_ref)))
# Get the betweenness values

vmap=betw_map_v.copy()

# Scale them for a more insightful visualization

vmap.a = 500 * (np.sqrt(vmap.a) + 0.005)

# Draw the graph

plot_graph(gt_graph, gt.fruchterman_reingold_layout(gt_graph), vmap=vmap)
pgrank_map=gt.pagerank(gt_graph)

pgrank_array=pgrank_map.a



plot_scatter(v_array, pgrank_array, 'pagerank', ylim=[min(pgrank_array)-1e-5,max(pgrank_array)+1e-5])



# Get one among the highest

high_pgrank_v=np.where(pgrank_array == pgrank_array.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_pgrank_v)]) 

print("Article with the highest PageRank centrality is: '%s'" % (get_title(paper_ref)))

if not (not highest_in_deg_v) and high_pgrank_v==highest_in_deg_v:

    print("Note that, coincidentally, this is the same paper as the one with the highest prestige as determined by the in-degree centrality above.")
hits_eig, auth_map_v, hubs_map_v=gt.hits(gt_graph)



auth_array_v=auth_map_v.a

hubs_array_v=hubs_map_v.a



fig = plt.figure(figsize=(8, 5.5))

    

plt.scatter(v_array, auth_array_v, color="#004080", edgecolors='black', alpha=0.75, label='hubs') 

plt.scatter(v_array, hubs_array_v, color="gray", edgecolors='black', alpha=0.75, label='authorities') 



# Unomment the line below to limit th y-axis values between 0.0 and 0.1

plt.ylim([0.0,0.1])



# Apply a few other plot settings

plt.xticks(fontsize=14)  

plt.yticks(fontsize=14)  

plt.xlabel('node ids', fontsize=16)  

plt.ylabel('authorities and hubs', fontsize=16)   

ax = plt.gca()  

ax.get_xaxis().tick_bottom()  

ax.get_yaxis().tick_left() 

plt.legend()

plt.show()



# Look at highest values

high_auth_v=np.where(auth_array_v == auth_array_v.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_auth_v)]) 

print("Article with the highest HITS-authority centrality is: '%s'" % (get_title(paper_ref)))

if not (not highest_in_deg_v) and high_auth_v==highest_in_deg_v:

    print("Note that, coincidentally, this is the same paper as the one with the highest prestige as determined by the in-degree centrality above.")

    

high_hubs_v=np.where(hubs_array_v == hubs_array_v.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_hubs_v)]) 

print("Article with the highest HITS-hubs centrality is: '%s'" % (get_title(paper_ref)))

if not (not highest_out_deg_v) and high_hubs_v==highest_out_deg_v:

    print("Note that, coincidentally, this is the same paper as the one with the highest prestige as determined by the out-degree centrality above.")
loc_clust_map=gt.local_clustering(gt_graph)

loc_clust_array_v=loc_clust_map.a



plot_hist(loc_clust_array_v, 'local clustering coefficient', scale="log", xlim=[0.01,None])



# Get one among the highest

high_loc_clust_v=np.where(loc_clust_array_v == loc_clust_array_v.max())[0][0]

paper_ref = rdflib.URIRef(vnames[str(high_loc_clust_v)]) 

print("Article with the highest local clustering is: '%s'" % (get_title(paper_ref)))
# Again, we can plot a neat visualization to see where the local clustering is high

# Get the clustering values

vmap=loc_clust_map.copy()

# Scale them slightly for a more insightful visualization

vmap.a = 10 * (np.sqrt(vmap.a) + 0.25)

# Draw the graph

plot_graph(gt_graph, gt.sfdp_layout(gt_graph), vmap=vmap)
glob_clust_mean=gt.global_clustering(gt_graph)

print("Global clustering coefficient: %f +- %f" % glob_clust_mean)
glob_assort_in_mean=gt.assortativity(gt_graph, "out")

print("Assortativity of out-degrees: %.5f +- %.5f" % glob_assort_in_mean)

glob_assort_out_mean=gt.assortativity(gt_graph, "in")

print("Assortativity of in-degrees: %.5f +- %.5f" % glob_assort_out_mean)

glob_assort_total_mean=gt.assortativity(gt_graph, "total")

print("Assortativity of total degrees: %.5f +- %.5f" % glob_assort_total_mean)
# First, we query the index of the node that corresponds to the paper of interest

paper="http://dx.doi.org/10.1016/j.jcv.2008.04.002"

node_id=vnames_inv[paper]

# Now we simply draw the betweenness value at the array index = node_id

paper_betw=betw_array_v[node_id]

# And evaluate the value of this betweenness with respect to the mean of the entire betweenness array

print("Evaluated with respect to the mean: %s" % (paper_betw/(np.mean(betw_array_v))))

# However, as can be seen from the plot in Sec. 1.3, the betweenness values are considerably skewed (i.e. most are close to zero while a few have high values). 

# Thus, it would be more appropriate to evaluate the value of this betweenness with respect to the median of the entire betweenness array. However, the median is zero.

print(np.median(betw_array_v))

# Therefore, we compare it to the highest betweenness value

print("Betweenness centrality of the considered paper: %s" % paper_betw)

print("Highest betweenness centrality value within the network: %s" % betw_array_v.max())

# Thus, the betweenness centrality of the considered paper is higher than that of most papers in the network but by one order of magnitude lower than the highest betweenness value

print("Their quotient: %s" % (paper_betw/betw_array_v.max()))