# keras modules 
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Reshape, Dropout, Dense, Input, Concatenate
from keras.models import Sequential, Model

# sklearn modules 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans 
from sklearn import preprocessing 
import matplotlib.pyplot as plt

# other python utilities 
from collections import Counter 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import warnings, math
import gc 

# ignore the warnings 
warnings.filterwarnings('ignore')
path = "../input/io/"
donors_df = pd.read_csv(path+"Donors.csv")
donations_df = pd.read_csv(path+"Donations.csv")
schools_df = pd.read_csv(path+"Schools.csv")
projects_df = pd.read_csv(path+"Projects.csv")
## Merge donations and donors 
donation_donor = donations_df.merge(donors_df, on='Donor ID', how='inner')

## Merge projects and schools 
projects_df = projects_df.merge(schools_df, on='School ID', how='inner')

## Create some additional features in projects data
projects_df['cost'] = projects_df['Project Cost']\
                      .apply(lambda x : float(str(x).replace("$","").replace(",","")))
projects_df['Posted Date'] = pd.to_datetime(projects_df['Project Posted Date'])
projects_df['Posted Year'] = projects_df['Posted Date'].dt.year
projects_df['Posted Month'] = projects_df['Posted Date'].dt.month

## Merge projects and donations (and donors)
master_df = projects_df.merge(donation_donor, on='Project ID', how='inner')

## Delete unusued datasets and clear the memory
del donation_donor, schools_df
gc.collect()
## Create a smaller version of data so that it runs on kaggle kernel
## keep only fully funded projects
projects_mini = projects_df[projects_df['Project Current Status'] == "Fully Funded"]

## Set rows = -1 to run on complete dataset, To run in kaggle kernel, I am setting to a smaller number 
rows = 5000

## keep only the projects of 2017, quarter 3, take small sample, (so that it runs on kaggle kernels)
if rows != -1:
    projects_mini = projects_mini[(projects_mini['Posted Year'] == 2017) &
                                  (projects_mini['Posted Month'] > 9)]
    projects_mini = projects_mini.reset_index()[:rows]

## replace the missing values and obtain project essay values 
projects_mini['Project Essay'] = projects_mini['Project Essay'].fillna(" ")
xtrain = projects_mini['Project Essay'].values
EMBEDDING_FILE = '../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec'

embeddings_index = {}
f = open(EMBEDDING_FILE, encoding="utf8")
count = 0
for line in tqdm(f):
    count += 1
    ## Remove this if condition to read 2M rows 
    if count == 500000: 
        break
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
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

xtrain_embeddings = [generate_doc_vectors(x) for x in tqdm(xtrain)]

del xtrain
gc.collect()
scl = preprocessing.StandardScaler()
xtrain_embeddings = np.array(xtrain_embeddings)
xtrain_embeddings = scl.fit_transform(xtrain_embeddings)
projects_mini[['Project ID', 'Project Title', 'Project Subject Category Tree', 'Project Resource Category', 'Project Cost']].head()
print ("Project ID: " + projects_mini['Project ID'].iloc(0)[0])
print ("Project Vector: ")
print (xtrain_embeddings)
# slicing the users dataset so that it runs in kaggle kernel memory 
users = master_df[(master_df['Donor Is Teacher'] == 'No') & 
                  (master_df['Donor State'] == 'Texas') & 
                  (master_df['Posted Year'] == 2017)].reset_index()

users1 = users[:1000]
del master_df
gc.collect()
## load the external dataset
external_context = pd.read_csv('../input/external-context/area_context_texas.csv')
features = list(set(external_context.columns) - set(['id', 'zipcode']))
agg_doc = {}
for feat in features:
    agg_doc[feat] = 'mean'

area_context = external_context.groupby('id').agg(agg_doc).reset_index().rename(columns = {'id' : 'Donor Zip'})
area_context = area_context[area_context['Housing Units'] != 0]
area_context.head()
features = list(set(area_context.columns) - set(['Donor Zip']))
inretia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(area_context[features])
    inretia.append(kmeans.inertia_)
plt.plot(range(1,11),inretia)
plt.title('Finding the Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.xlabel('Kmeans Inretia')
plt.show()
# apply kmeans clustering 
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0) 
area_context['area_context_cluster'] = kmeans.fit_predict(area_context[features])

# merge with the donors data
users1['Donor Zip'] = users1['Donor Zip'].astype(str)
area_context['Donor Zip'] = area_context['Donor Zip'].astype(str)

users1 = users1.merge(area_context[['Donor Zip', 'area_context_cluster']], on="Donor Zip", how="left")
area_context[['Donor Zip', 'area_context_cluster']].head(10)
users1['Project Essay'] = users1['Project Essay'].fillna(" ")
utrain = users1['Project Essay'].values

utrain_embeddings = [generate_doc_vectors(x) for x in tqdm(utrain)]
utrain_embeddings = np.array(utrain_embeddings)
utrain_embeddings = scl.fit_transform(utrain_embeddings)

del utrain
gc.collect()
## handle few missing values 
users1['Project Type'] = users1['Project Type'].fillna("Teacher Led")
users1['Project Subject Category Tree'] = users1['Project Subject Category Tree'].fillna(" ")
users1['area_context_cluster'] = users1['area_context_cluster'].astype(str)

## aggregate the donors and their past donations in order to create their donor - profiles
user_profile = users1.groupby('Donor ID').agg({'Donation Amount' : ['min', 'max', 'mean', 'median'],
                                               'cost' : ['min', 'max', 'mean', 'median'], 
                                      'Project Subject Category Tree' : lambda x: ", ".join(x), 
                                      'Project ID' : lambda x: ",".join(x), 
                                      'School Metro Type' : lambda x: ",".join(x), 
                                      'Project Title' : lambda x: ",".join(x), 
                                      'area_context_cluster' : lambda x: ",".join(x), 
                                      'School Percentage Free Lunch' : 'mean',
                                      'Project Grade Level Category' : lambda x : ",".join(x),
                                      'Project Type' : 'count'}
                                    ).reset_index().rename(columns={'Project Type' : "Projects Funded"})
## flatten the features of every donor

def get_map_features(long_text):
    a = long_text.split(",")
    a = [_.strip() for _ in a]
    mapp = dict(Counter(a))
    return mapp
    
user_profile['Category_Map'] = user_profile['Project Subject Category Tree']['<lambda>'].apply(get_map_features)
user_profile['Projects_Funded'] = user_profile['Project ID']['<lambda>'].apply(get_map_features)
user_profile['GradeLevel_Map'] = user_profile['Project Grade Level Category']['<lambda>'].apply(get_map_features)
user_profile['AreaContext_Map'] = user_profile['area_context_cluster']['<lambda>'].apply(get_map_features)
user_profile['SchoolMetroType_Map'] = user_profile['School Metro Type']['<lambda>'].apply(get_map_features)
user_profile = user_profile.drop(['Project Grade Level Category', 'Project Subject Category Tree',  'School Metro Type', 'Project ID', 'area_context_cluster'], axis=1)

user_profile.head()
def get_average_vector(project_ids):
    ids = list(project_ids.keys())
    
    donor_proj_vec = []
    for idd in ids:        
        unique_proj_ids = users1[users1['Project ID'] == idd].index.tolist()[0]
        donor_proj_vec.append(utrain_embeddings[unique_proj_ids])
    proj_vec = np.array(donor_proj_vec).mean(axis=0)
    return proj_vec 

user_profile['project_vectors'] = user_profile['Projects_Funded'].apply(lambda x : get_average_vector(x))
user_profile[['Donor ID', 'project_vectors']].head(10)
# compute the project project interactions 
project_interactions = linear_kernel(xtrain_embeddings, xtrain_embeddings)
# create the edges of one node with other most similar nodes  
project_edges = {}
for idx, row in projects_mini.iterrows():
    similar_indices = project_interactions[idx].argsort()[:-100:-1]
    similar_items = [(project_interactions[idx][i], projects_mini['Project ID'][i]) for i in similar_indices]
    project_edges[row['Project ID']] = similar_items[:20]
def get_project(id):
    return projects_mini.loc[projects_mini['Project ID'] == id]['Project Title'].tolist()[0]

def similar_projects(project_id, num):
    print("Project: " + get_project(project_id))
    print("")
    print("Similar Projects: ")
    print("")
    recs = project_edges[project_id][1:num]
    for rec in recs:
        print(get_project(rec[1]) + " (score:" + str(rec[0]) + ")")
similar_projects(project_id="a0446d393feaadbeb32cd5c3b2b36d45", num=10)
similar_projects(project_id="83b4f3fbe743cb12ae2be7347ef03ecb", num=10)
user_embeddings = user_profile['project_vectors'].values

user_embeddings_matrix = np.zeros(shape=(user_embeddings.shape[0], 300))
for i,embedding in enumerate(user_embeddings):
    user_embeddings_matrix[i] = embedding

donors_interactions = linear_kernel(user_embeddings_matrix, user_embeddings_matrix)
user_edges = {}
for idx, row in user_profile.iterrows():
    similar_indices = donors_interactions[idx].argsort()[:-10:-1]

    similar_items = [(float(donors_interactions[idx][i]), list(user_profile['Donor ID'])[i]) for i in similar_indices]
    user_edges[row['Donor ID'][0]] = similar_items[1:]
def get_donor(id):
    return user_profile.loc[user_profile['Donor ID'] == id]['Donor ID'].tolist()[0]

def similar_users(donor_id, num):
    print("Donor: " + get_donor(donor_id))
    print ("Projects: " + str(user_profile[user_profile['Donor ID'] == donor_id]['Project Title']['<lambda>'].iloc(0)[0]))

    print("")
    print("Similar Donors: ")
    print("")    
    recs = user_edges[donor_id][:num]
    for rec in recs:
        print("DonorID: " + get_donor(rec[1]) +" | Score: "+ str(rec[0]) )
        print ("Projects: " + str(user_profile[user_profile['Donor ID'] == rec[1]]['Project Title']['<lambda>'].iloc(0)[0]))
        print   ("")
similar_users(donor_id="fee882faa77bc6691bd24d4d5abd5733", num=5)
similar_users(donor_id="d52242e9d5006fb97fcdb5565982f0ad", num=5)
class DonorsGraph():
    """
    Class to create the graph for donors and save their information in different nodes.
    """
    
    
    def __init__(self, graph_name):
        self.graph = {}
        self.graph_name = graph_name
    
    # function to add new nodes in the graph
    def _create_node(self, node_id, node_properties):
        self.graph[node_id] = node_properties 
    
    # function to view the nodes in the graph
    def _view_nodes(self):
        return self.graph
    
    # function to create edges
    def _create_edges(self, node_id, node_edges):
        if node_id in self.graph:
            self.graph[node_id]['edges'] = node_edges
## initialize the donors graph
dg = DonorsGraph(graph_name = 'donor')

## iterate in donor profiles and add the nodes
for idx, row in user_profile.iterrows():
    node_id = row['Donor ID'].tolist()[0]
    node_properties = dict(row)
    dg._create_node(node_id, node_properties)
node = dg._view_nodes()['12d74c3cd5f21ed4b17c781da828d076']
node[('project_vectors','')][0:50]
del node[('project_vectors','')]
node
def get_donor(id):
    return user_profile.loc[user_profile['Donor ID'] == id]['Donor ID'].tolist()[0]

def get_similar_donors(donor_id, num):
    # improve this algorithm - > currently only text, add other features as well 
    recs = user_edges[donor_id][:num]    
    return recs 

for idx, row in user_profile.iterrows():
    node_id = row['Donor ID'].tolist()[0]
    node_edges = get_similar_donors(donor_id=node_id, num=5)
    dg._create_edges(node_id, node_edges)
dg._view_nodes()['00b3c149822c79e4fca9be0bea5c900c']['edges']
class ProjectsGraph():
    def __init__(self, graph_name):
        self.graph = {}
        self.graph_name = graph_name
        
    def _create_node(self, node_id, node_properties):
        self.graph[node_id] = node_properties 
    
    def _view_nodes(self):
        return self.graph
    
    def _create_edges(self, node_id, node_edges):
        if node_id in self.graph:
            self.graph[node_id]['edges'] = node_edges
pg = ProjectsGraph(graph_name = 'projects')

for idx, row in projects_mini.iterrows():
    node_id = row['Project ID']
    node_properties = dict(row)
    del node_properties['Project Essay']
    del node_properties['Project Need Statement'] 
    del node_properties['Project Short Description']
    pg._create_node(node_id, node_properties)

pg._view_nodes()['83b4f3fbe743cb12ae2be7347ef03ecb']
def get_similar_projects(project_id, num):
    recs = project_edges[project_id][:num]
    return recs 

for idx, row in projects_mini.iterrows():
    node_id = row['Project ID']
    node_edges = get_similar_projects(project_id=node_id, num=5)
    pg._create_edges(node_id, node_edges)
pg._view_nodes()['83b4f3fbe743cb12ae2be7347ef03ecb']['edges']
from IPython.core.display import display, HTML, Javascript
import IPython.display
import json

nodes = []
links = []

nodes.append({'id' : 'Donors', 'group' : 2, 'size' : 20 })
for key, val in list(dg._view_nodes().items())[:50]:
    if len(val['edges']) == 0:
        continue
    nodes.append({'id' : key, 'group' : 2, 'size' : 15})
    links.append({"source" : "Donors", "target" : key, "value" : 10})
    
    for node in val['edges']:
        nodes.append({'id' : node[1], 'group' : 2, 'size' : 12})
        
        sv = np.log(node[0])
        ew = 10
        if sv > 6:
            ew = 100
        elif sv > 5:
            ew = 20
        elif sv > 4:
            ew = 15
        else:
            ew = 10
                    
        links.append({"source": key, "target": node[1], "value": ew})
doc = {'nodes' : nodes, 'links' : links}
with open("donorg.json", "w") as fout:
    fout.write(json.dumps(doc))
    
    

nodes = []
links = []
nodes.append({'id' : 'Projects', 'group' : 0, 'size' : 20, "title" : "Projects" })
for key, val in list(pg._view_nodes().items())[:75]:
    if len(val['edges']) == 0:
        continue

    nodes.append({'id' : key, 'group' : 0, 'size' : 15})
    links.append({"source" : "Projects","title" : val['Project Title'], "target" : key, "value" : 10})
    for node in val['edges']:
        title = projects_mini[projects_mini['Project ID'] == node[1]]['Project Title'].iloc(0)[0]
        nodes.append({'id' : node[1], 'group' : 2, 'size' : 12, "title": title})
        links.append({"source": key, "target": node[1], "value": 8})
doc = {'nodes' : nodes, 'links' : links}
with open("projectg.json", "w") as fout:
    fout.write(json.dumps(doc))
html7="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 250px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="donorg" width="860" height="760"></svg>"""

js7="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
 
require(["d3"], function(d3) {// Dimensions of sunburst.
 
var svg = d3.select("#donorg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(20).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("donorg.json", function(error, graph) {
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
            div.html(d.id)
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    

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

h = display(HTML(html7))
j = IPython.display.Javascript(js7)
IPython.display.display_javascript(j)

html8="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 250px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="projectg" width="860" height="760"></svg>"""

js8="""
 
require(["d3"], function(d3) {// Dimensions of sunburst.
 
var svg = d3.select("#projectg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(20).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("projectg.json", function(error, graph) {
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
            div.html(d.title)
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    

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

h = display(HTML(html8))
j = IPython.display.Javascript(js8)
IPython.display.display_javascript(j)

def connect_project_donors(project_id):
        
    # get the project index
    proj_row = projects_mini[projects_mini['Project ID'] == project_id]
    proj_ind = proj_row.index

    # get the project vector
    proj_vector = xtrain_embeddings[proj_ind]
    
    # match the vector with the user vectors 
    cossim_proj_user = linear_kernel(proj_vector, user_embeddings_matrix)
    reverse_matrix = cossim_proj_user.T
    reverse_matrix = np.array([x[0] for x in reverse_matrix])
    similar_indices = reverse_matrix.argsort()[::-1]
    
    # filter the recommendations
    projects_similarity = []
    recommendations = []
    top_users = [(reverse_matrix[i], user_profile['Donor ID'][i]) for i in similar_indices[:10]]
    for x in top_users:
        user_id = x[1]
        user_row= user_profile[user_profile['Donor ID'] == user_id]
        
        ## to get the appropriate recommendations, filter them using other features 
        cat_count = 0
        
        ## Making use of Non Text Features to filter the recommendations
        subject_categories = proj_row['Project Subject Category Tree'].iloc(0)[0]
        for sub_cat in subject_categories.split(","):
            if sub_cat.strip() in user_row['Category_Map'].iloc(0)[0]:
                cat_count += user_row['Category_Map'].iloc(0)[0][sub_cat.strip()]

        grade_category = proj_row['Project Grade Level Category'].iloc(0)[0]
        if grade_category in user_row['Category_Map'].iloc(0)[0]:
            cat_count += user_row['Category_Map'].iloc(0)[0][grade_category]

        metro_type = proj_row['School Metro Type'].iloc(0)[0]
        if metro_type in user_row['SchoolMetroType_Map'].iloc(0)[0]:
            cat_count += user_row['SchoolMetroType_Map'].iloc(0)[0][metro_type]
        
        x = list(x)
        x.append(cat_count)
        recommendations.append(x)
        
        ## Find similar donors
        donor_nodes = dg._view_nodes()
        if x[1] in donor_nodes:
            recommendations.extend(donor_nodes[x[1]]['edges'])

    ## Find Similar Projects 
    project_nodes = pg._view_nodes()
    if project_id in project_nodes:
        projects_similarity.extend(project_nodes[project_id]['edges'])    

    return projects_similarity, recommendations
    
def get_recommended_donors(project_id):
    # Find the recommended donors and the similar projects for the given project ID 
    sim_projs, recommendations = connect_project_donors(project_id)

    # filter the donors who have already donated in the project
    current_donors = donations_df[donations_df['Project ID'] == project_id]['Donor ID'].tolist()

    # Add the donors of similar projects in the recommendation
    for simproj in sim_projs:
        recommendations.extend(connect_project_donors(simproj[1])[1])
    
    ######## Create final recommended donors dataframe 
    # 1. Most relevant donors for a project 
    # 2. Similar donors of the relevant donors 
    # 3. Donors of the similar project 
    
    recommended_df = pd.DataFrame()
    recommended_df['Donors'] = [x[1] for x in recommendations]
    recommended_df['Score'] = [x[0] for x in recommendations]
    recommended_df = recommended_df.sort_values('Score', ascending = False)
    recommended_df = recommended_df.drop_duplicates()

    recommended_df = recommended_df[~recommended_df['Donors'].isin(current_donors)]
    return recommended_df

def _get_results(project_id):
    proj = projects_mini[projects_mini['Project ID'] == project_id]
    print ("Project ID: " + project_id )
    print ("Project Title: " + proj['Project Title'].iloc(0)[0])
    print ("")

    print ("Recommended Donors: ")
    recs = get_recommended_donors(project_id)
    donated_projects = []
    for i, row in recs.head(10).iterrows():
        donor_id = row['Donors']
        print (donor_id +" | "+ str(row['Score']))
        donor_projs = user_profile[user_profile['Donor ID'] == donor_id]['Project Title']['<lambda>'].iloc(0)[0]
        donor_projs = donor_projs.split(",")
        for donor_proj in donor_projs:
            if donor_proj not in donated_projects:
                donated_projects.append(donor_proj)
    print ("")
    print ("Previous Projects of the Recommended Donors: ")
    for proj in donated_projects:
        print ("-> " + proj)

project_id = "d20ec20b8e4de165476dfd0a68da0072"
_get_results(project_id)
project_id = "ad51c22f0d31c7dc3103294bdd7fc9c1"
_get_results(project_id)
## create the interaction data frames
interactions = users[['Project ID', 'Donor ID', 'Donation Amount']]
interactions['interaction_score_2'] = np.log(interactions['Donation Amount'])
unique_donors = list(interactions['Donor ID'].value_counts().index)
donor_map = {}
for i, user in enumerate(unique_donors):
    donor_map[user] = i+1

unique_projs = list(interactions['Project ID'].value_counts().index)
proj_map = {}
for i, proj in enumerate(unique_projs):
    proj_map[proj] = i+1

tags = {'donors' : donor_map, 'projects' : proj_map}
def getID(val, tag):
    return tags[tag][val]
     
interactions['proj_id'] = interactions['Donor ID'].apply(lambda x : getID(x, 'donors'))
interactions['user_id'] = interactions['Project ID'].apply(lambda x : getID(x, 'projects'))
# remove the duplicate entries in the dataset 
max_userid = interactions['user_id'].drop_duplicates().max() + 1
max_movieid = interactions['proj_id'].drop_duplicates().max() + 1
shuffled_interactions = interactions.sample(frac=1., random_state=153)
PastDonors = shuffled_interactions['user_id'].values
PastProjects = shuffled_interactions['proj_id'].values
Interactions = shuffled_interactions['interaction_score_2'].values
def create_model(n_donors, m_projects, embedding_size):
    
    # add input layers for donors and projects
    donor_id_input = Input(shape=[1], name='donor')
    project_id_input = Input(shape=[1], name='project')

    # create donor and project embedding layers 
    donor_embedding = Embedding(output_dim=embedding_size, input_dim=n_donors,
                               input_length=1, name='donor_embedding')(donor_id_input)
    project_embedding = Embedding(output_dim=embedding_size, input_dim=m_projects,
                               input_length=1, name='project_embedding')(project_id_input)
    
    # perform reshaping on donor and project vectors 
    donor_vecs = Reshape([embedding_size])(donor_embedding)
    project_vecs = Reshape([embedding_size])(project_embedding)
    
    # concatenate the donor and project embedding vectors 
    input_vecs = Concatenate()([donor_vecs, project_vecs])
    
    # add a dense layer
    x = Dense(128, activation='relu')(input_vecs)
    
    # add the output layer
    y = Dense(1)(x)
    
    # create the model using inputs and outputs 
    model = Model(inputs=[donor_id_input, project_id_input], outputs=y)
    
    # compile the model, add optimizer function and loss function
    model.compile(optimizer='adam', loss='mse')  
    return model
embedding_size = 10
model = create_model(max_userid, max_movieid, embedding_size)
model.summary()
def rate(model, user_id, item_id):
    return model.predict([np.array([user_id]), np.array([item_id])])[0][0]

def predict_rating(movieid, userid):
    return rate(model, movieid - 1, userid - 1)
## with more data, nb_epooch can also be increased
history = model.fit([PastDonors, PastProjects], Interactions, nb_epoch=2, validation_split=.20)
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
Past_Donors = users[['Donor ID', 'Donor City']]
Past_Projects = users[['Project ID', 'Project Title']]

Past_Donors = Past_Donors.drop_duplicates()
Past_Projects = Past_Projects.drop_duplicates()

Past_Donors['user_id'] = Past_Donors['Donor ID'].apply(lambda x : getID(x, 'donors'))
Past_Projects['proj_id'] = Past_Projects['Project ID'].apply(lambda x : getID(x, 'projects'))

## for this sample run, get common IDs from content based and 
## collaborative approaches to get the results together

list1 = list(projects_mini['Project ID'].values)
list2 = list(Past_Projects['Project ID'].values)
common_ids = list(set(list1).intersection(list2))
idd = proj_map['ad51c22f0d31c7dc3103294bdd7fc9c1']

user_ratings = interactions[interactions['proj_id'] == idd][['user_id', 'proj_id', 'interaction_score_2']]
user_ratings['predicted_amt'] = user_ratings.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations = interactions[interactions['user_id'].isin(user_ratings['user_id']) == False][['user_id']].drop_duplicates()
recommendations['predicted_amt'] = recommendations.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations['predicted_amt'] = np.exp(recommendations['predicted_amt'])
recommendations.sort_values(by='predicted_amt', ascending=False).merge(Past_Donors, on='user_id', how='inner', suffixes=['_u', '_m']).head(10)
project_id = "ad51c22f0d31c7dc3103294bdd7fc9c1"
_get_results(project_id)
title = projects_mini[projects_mini['Project ID'] == project_id]['Project Title'].iloc(0)[0]
print ("Project ID: " + project_id )
print ("Project Title: " + title)
print ("")
    
idd = proj_map[project_id]
user_ratings = interactions[interactions['proj_id'] == idd][['user_id', 'proj_id', 'interaction_score_2']]
user_ratings['predicted_amt'] = user_ratings.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations = interactions[interactions['user_id'].isin(user_ratings['user_id']) == False][['user_id']].drop_duplicates()
recommendations['predicted_amt'] = recommendations.apply(lambda x: predict_rating(idd, x['user_id']), axis=1)
recommendations['predicted_amt'] = np.exp(recommendations['predicted_amt'])
recs = recommendations.sort_values(by='predicted_amt', ascending=False).merge(Past_Donors, on='user_id', how='inner', suffixes=['_u', '_m']).head(10)

past_projs = []
print ("Donors based on Behavioural Similarity: ")
for i, row in recs.head(5).iterrows():
    print (row['Donor ID'] +" | Donated Amount: "+ str(row['predicted_amt']))
    dons = donations_df[donations_df['Donor ID'] == row['Donor ID']]
    for i,x in dons.head(3).iterrows():
        projs = projects_df[projects_df['Project ID'] == x['Project ID']]
        title = projs['Project Title'].iloc(0)[0]
        cost = projs['Project Cost'].iloc(0)[0]
        txt =title + " ($" + str(cost) + ")"
        past_projs.append(txt)

print ("")
print ("Past Projects and Donations: ")
for proj_ in past_projs:
    print (proj_)

