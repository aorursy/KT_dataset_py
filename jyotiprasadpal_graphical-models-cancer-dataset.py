from IPython.display import Image
# Image('../input/cancer_model/cancer_model.png')
!pip install pgmpy
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from pgmpy.models import BayesianModel

from pgmpy.estimators import MaximumLikelihoodEstimator

from pgmpy.inference import VariableElimination

import networkx as nx
data = pd.read_csv('../input/cancer-dataset/cancer_data.csv')

data.head()
data.replace({'yes': 1, 'no': 0}, inplace=True)

data.head()
# Create a bayesian model with nodes and edges.

model = BayesianModel([('asia', 'tub'), ('tub', 'either'), ('either', 'xray'), ('either', 'dysp'), ('smoke', 'lung'), ('lung', 'either'), ('smoke', 'bronc'), ('bronc', 'dysp')])

# Estimate the CPD for each variable based on a given data set.

model.fit(data, estimator=MaximumLikelihoodEstimator)



fig, ax = plt.subplots(figsize=(9, 7))

# https://stackoverflow.com/questions/54019644/how-to-display-a-network-with-better-separation-using-the-networkx-package/54021354#54021354

# pos = nx.spring_layout(model) 

positions={'asia':(1,8), 'tub':(2,5), 'either':(3,3), 'xray':(3,0), 'smoke':(5,8), 'lung':(4,6), 'bronc':(6,3), 'dysp':(6,0), }

nx.draw(model, pos=positions, with_labels=True, node_size = 4000, font_size = 20, arrowsize=20, node_color='red', ax=ax)
# Check the model for various errors. This method checks for the following errors.

# * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).

# * Checks if the CPDs associated with nodes are consistent with their parents.

model.check_model()
# Get the CPDs of the nodes

model.get_cpds()
print(model.get_cpds('smoke'))

print(model.get_cpds('tub'))
# Active trail: For any two variables A and B in a network if any change in A influences the values of B then we say

# that there is an active trail between A and B.

# In pgmpy active_trail_nodes gives a set of nodes which are affected (i.e. correlated) by any 

# change in the node passed in the argument.

print(model.active_trail_nodes('smoke', observed=['bronc']))

print(model.is_active_trail(start='smoke', end='xray', observed=['either']))
#Create Inference Object of the model and run VE algorithm on it

inference = VariableElimination(model)



# Query1: Will the patient be having ‘tuberculosis’ given that the patient smokes but no problem of dyspnoea?

query1= inference.map_query(variables=['tub'], evidence={'smoke': 1, 'dysp': 0})

print(query1)



# Query2: What is the most probable value of ‘xray’ conditioning patient is from asia and has lung cancer?

query2= inference.map_query(variables=['xray'], evidence={'asia': 1, 'lung': 1})

print(query2)



# Query 3: What is the most probable value of bronc if patient has the issue of dyspnoea , he is from asia but does not have either of the diseases?

query3= inference.map_query(variables=['bronc'], evidence={'dysp': 1, 'asia': 1, 'either': 0})

print(query3)
#write data to csv

data = {'Query1': [[key for key in query1.keys()][0], [value for value in query1.values()][0]], 

        'Query2': [[key for key in query2.keys()][0], [value for value in query2.values()][0]], 

        'Query3': [[key for key in query3.keys()][0], [value for value in query3.values()][0]]

       }

result = pd.DataFrame(data) 

result.to_csv('output.csv', index=False)

result