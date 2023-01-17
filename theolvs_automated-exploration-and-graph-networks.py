# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

# Matplotlib default configuration
plt.style.use('ggplot')
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = "#d4d4d4"
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2
class DataExplorer(object):
    def __init__(self,data):
        self.data = data
        print(f"Shape : {self.data.shape}")
        
    def _repr_html_(self):
        return self.data.head()._repr_html_()
    
    #-----------------------------------------------------------------------------------------------
    # Exploration
    
    def explore(self):
        for var in self.data.columns:
            self.explore_column(var)
    
    def explore_column(self,var,max_rows = 10000,threshold_occurences = 0.5,**kwargs):
        print(f">> Exploration of {var}")
        column = self.data[var]
        dtype = column.dtype
        
        if len(column) > max_rows:
            column = column.sample(max_rows)
            
        if dtype == np.float64:
            self.show_distribution(var = var,column = column,**kwargs)
        else:
            if len(column.unique()) / len(column) < threshold_occurences:
                self.show_top_occurences(var = var,column = column,**kwargs)
            else:
                print(f"... Too many occurences, '{var}' is probably an ID")
                
        print("")
    
    #-----------------------------------------------------------------------------------------------
    # Visualizations
    
    def show_distribution(self,var = None,column = None,figsize = (15,4),kind = "hist"):
        if column is None:
            column = self.data[var]
        column.plot(kind = kind,figsize = figsize)
        if var is not None:
            plt.title(var)
        plt.show()
        
    def show_top_occurences(self,var = None,column = None,n = 30,figsize = (15,4),kind = "bar"):
        if column is None:
            column = self.data[var]
            
        column.value_counts().head(n).plot(kind = kind,figsize = figsize)
        if var is not None:
            plt.title(var)
        plt.show()
resources = pd.read_csv("../input/Resources.csv")
schools = pd.read_csv("../input/Schools.csv")
donors = pd.read_csv("../input/Donors.csv")
donations = pd.read_csv("../input/Donations.csv")
teachers = pd.read_csv("../input/Teachers.csv")
projects = pd.read_csv("../input/Projects.csv")
explorer = DataExplorer(resources)
explorer
explorer.explore()
explorer = DataExplorer(schools)
explorer
explorer.explore()
explorer = DataExplorer(donors)
explorer
explorer.explore()
explorer = DataExplorer(donations)
explorer
explorer.explore()
explorer = DataExplorer(teachers)
explorer
explorer.explore()
explorer = DataExplorer(projects)
explorer
explorer.explore()
import networkx as nx

G = nx.Graph()
columns = { file:list(eval(file).columns) for file in ["resources","schools","donors","donations","teachers","projects"]}
for file_i in columns:
    for file_j in columns:
        if file_i != file_j:
            intersection = set(columns[file_i]).intersection(set(columns[file_j]))
            if len(intersection) >= 1:
                G.add_edge(file_i,file_j,intersection=intersection)
plt.figure(figsize = (10,10))
pos = nx.spring_layout(G)
nx.draw(G,pos = pos)
_ = nx.draw_networkx_labels(G,pos,font_weight="bold")
_ = nx.draw_networkx_edge_labels(G,pos)
plt.show()
network = nx.Graph()
donations["Donation Received Date"] = pd.to_datetime(donations["Donation Received Date"])
donations.set_index("Donation Received Date",inplace = True)
subset_donations = donations["2018-01"]
subset_donations.shape
for i,row in subset_donations.iterrows():
    donor = row["Donor ID"]
    project = row["Project ID"]
    amount = row["Donation Amount"]
    network.add_node(donor,type = "donor")
    network.add_node(project,type = "project")
    network.add_edge(donor,project,amount = amount)
print(nx.info(network))
degree = pd.DataFrame(list(network.degree),columns = ["node_id","degree"]).set_index("node_id")
types = pd.DataFrame(pd.Series(nx.get_node_attributes(network,"type")),columns = ["node_type"])
degree = degree.join(types)
degree.head()
projects_degree = degree.query("node_type=='project'")
donors_degree = degree.query("node_type=='donor'")
(projects_degree["degree"].value_counts()/len(projects_degree)).head(10).plot(figsize = (15,4),label = "project")
(donors_degree["degree"].value_counts()/len(donors_degree)).head(10).plot(label = "donor")
plt.title("Distribution of degree in the network")
plt.xlabel("Node degree")
plt.ylabel("Number of nodes")
plt.legend()
plt.show()
from tqdm import tqdm_notebook
components = []
for i,component in enumerate(tqdm_notebook(nx.connected_component_subgraphs(network))):
    if len(component.nodes) > 2:
        components.append(component)
import tensorrec
interactions, user_features, item_features = tensorrec.util.generate_dummy_data(
    num_users=100,
    num_items=150,
    interaction_density=.05
)
interactions_df = pd.DataFrame(interactions.todense())
interactions_df.shape
user_features
item_features
# Build the model with default parameters
model = tensorrec.TensorRec()
# Fit the model for 5 epochs
model.fit(interactions, user_features, item_features, epochs=5, verbose=True)
# Predict scores and ranks for all users and all items
predictions = model.predict(user_features=user_features,
                            item_features=item_features)
predicted_ranks = model.predict_rank(user_features=user_features,
                                     item_features=item_features)
predictions.shape
predicted_ranks.shape