# install graph drawing and interaction libraries
# !pip uninstall networkx -y
# !pip install networkx==1.11

#pip install nxpd
# !pip install ipywidgets
# !apt-get install python-pydot

import pandas as pd
import numpy as np
import math

import sympy 
from sympy import *
from sympy import Matrix
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import Function

import networkx as nx

import __future__
import __init__

%matplotlib inline

import matplotlib.pylab as plt

import warnings
warnings.filterwarnings('ignore')

from IPython.display import Latex

#If you want all graphs to be drawn inline, then you can set a global parameter.
from nxpd import draw
from nxpd import nxpdParams
nxpdParams['show'] = 'ipynb'

import string

from ipywidgets import Image
#from IPython import display

from ipywidgets import interact
import ipywidgets as widgets

from IPython.display import Math, HTML, display




def create_Markov_matrix(height, width):
    '''
    randomly create a 'right' markov matrix (each row sums to 1)
    '''
    
    x = np.random.random(size=[width, height])

    # use float128 for precision when taking powers
    markov_matrix = x / x.sum(axis=0, dtype = np.float128)  

    markov_matrix = markov_matrix.transpose()
    
    return(markov_matrix)


def show_latex_matrix(matrix1, pre_latex = "", post_latex = ""):
    '''
    '''
    
    display(HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
               "latest.js?config=default'></script>"))

    return(Math(pre_latex + latex(Matrix(matrix1)) + post_latex))
    
    #return latex(Matrix(matrix1))
    #return(Latex('$%s$'%latex(Matrix(matrix1))))


def create_Markov_graph(markov_matrix, mapping = None):
    '''
    inputs:
    numpy matrix
    a mapping dictionary from integer node ids to anything
    
    returns:
    a networkx graph
    '''

    G=nx.from_numpy_matrix(markov_matrix, create_using=nx.DiGraph())

    # if no mapping is passed, just keeps the node ids
    if (mapping == None):
        mapping = dict(zip(list(range(markov_matrix.shape[0])), list(range(markov_matrix.shape[0]))))
        
    G=nx.relabel_nodes(G, mapping)

    nx.set_node_attributes(G, 'fontsize', 10)
    nx.set_node_attributes(G, 'color', 'black')

    nx.set_edge_attributes(G, 'fontsize', 10)
    nx.set_edge_attributes(G, 'color', 'grey')

    weights = nx.get_edge_attributes(G, 'weight')
    weights = nx.get_edge_attributes(G, 'weight')
    vals = weights.values()
    vals = np.array(list(vals)).astype(float)
    vals = np.round(vals, 2)
    weights = dict(zip(list(weights.keys()),(vals)))
    
    nx.set_edge_attributes(G, name='label', values=weights)
    
    return(G)


def compute_markov_states(markov_matrix, mapping = None, num_trials = 10):
    '''
    inputs:
    a numpy right markov matrix    
    a mapping dictionary from integer node ids to anything
    
    returns:
    list of all states of system
    '''

    states = []
    
    # if no mapping is passed, just keeps the node ids
    if (mapping == None):
        mapping = dict(zip(list(range(markov_matrix.shape[0])), list(range(markov_matrix.shape[0]))))
    
    height, width = markov_matrix.shape

    # set up horizontal cumulative sums for computing next state
    cum_sum = markov_matrix.cumsum(axis=1)

    # start state
    state_index = np.random.choice(list(range(height)))
    state = np.zeros(height)
    state[state_index] = 1

    states.append(mapping[state_index])    
    
    for i in range(num_trials):
        rand_num = np.random.rand()
        state_index = np.where(state == 1)[0][0]
        next_state = np.where(cum_sum[state_index,:] > rand_num)[0][0]
        state_index = next_state
        #print(mapping[state_index])
        #input()  #pause at each step
        states.append(mapping[state_index])
    return(states)


def create_alphabet_mapping_dict():
    '''
    map the node id numbers to letters
    '''
    
    # the node ids (let's use letters)
    alphabet = list(string.ascii_uppercase)
    keys = list(range(len(alphabet)))
    mapping = dict(zip(keys, alphabet))
    
    return mapping

def coarse_grain_markov(markov_matrix, timesteps):
    '''
    coarsegrain the Markov Matrix
    
    inputs:
    a right-markov matrix
    number of timesteps (powers of matrix)
    
    returns
    a right-markov matrix equal to the specified power of the input matrix
    '''
    
    return(np.linalg.matrix_power(markov_matrix, timesteps))
    
height = 2
width = 2

# map the node id numbers to letters
mapping = create_alphabet_mapping_dict()

# create the markov matrix
markov_matrix = create_Markov_matrix(height, width)

# display it nicely using latex
show_latex_matrix(markov_matrix, "T=")
# check that rows sum to 1
markov_matrix.sum(axis=1)
# create networkx graph
G = create_Markov_graph(markov_matrix, mapping)

# draw networkx graph with graphviz
im = draw(G, layout='circo')
im
num_trials = 10

compute_markov_states(markov_matrix, mapping, num_trials)
timesteps = 2

coarse_markov_2 = coarse_grain_markov(markov_matrix, timesteps)
show_latex_matrix(coarse_markov_2, pre_latex = "T^{" + str(timesteps) + "}=")
# check that each row sums to 1
coarse_markov_2.sum(axis = 1)
# create networkx graph
G = create_Markov_graph(coarse_markov_2, mapping)

# draw networkx graph with graphviz
im = draw(G, layout='circo')
im
timesteps = 100
coarse_markov = coarse_grain_markov(markov_matrix, timesteps)
show_latex_matrix(coarse_markov)
# check that each row sums to 1
coarse_markov.sum(axis = 1)
# create networkx graph
G = create_Markov_graph(coarse_markov, mapping)

# draw networkx graph with graphviz
im = draw(G, layout='circo')
im
G = nx.DiGraph()

G.add_edge("A", "B", label = '1-p_a')
G.add_edge("B", "A", label = 'p_a')
G.add_edge("A", "A", label = 'p_a')
G.add_edge("B", "B", label = '1-p_a')


# draw networkx graph with graphviz
draw(G, layout='dot')
G = nx.DiGraph()

G.add_edge("A", "B", label = 'p_b = 1-p_a')
G.add_edge("B", "A", label = 'p_a')
G.add_edge("A", "A", label = 'p_a')
G.add_edge("B", "B", label = 'p_b = 1-p_a')

# draw networkx graph with graphviz
draw(G, layout='dot')
# get the eigenvalues and eigenvectors
lambdas, eig_vectors = np.linalg.eig(markov_matrix.astype(np.float))

print('eigenvalues:', lambdas)
print()
print('eigenvectors:')
print(eig_vectors)
num_coarse_grainings = 10
lambda_values = np.zeros([num_coarse_grainings, height])
eigenvectors = []
p_A_A = [] #p(A|A)
p_B_B = [] #p(B|B)

for i in range(num_coarse_grainings):
    coarse_markov = coarse_grain_markov(markov_matrix, i)
    
    lambdas, eig_vectors = np.linalg.eig(coarse_markov.astype(np.float))
    lambda_values[i] = lambdas
    eigenvectors.append(eig_vectors)
    p_A_A.append(coarse_markov[0,0])
    p_B_B.append(coarse_markov[1,1])
    #print('$\lambdas$:',lambdas)
    #print('v',eig_vectors)
    #print()
# here plot the time evolution of the eigenvalues

col_names = []
for i in range(lambda_values.shape[1]):
    col_names.append('lambda_' + str(i + 1))
    
df = pd.DataFrame(lambda_values, columns=col_names)

df.plot();
plt.xlabel('time coarse graining')
plt.ylabel('eigenvalues');
# the phase diagram

plt.figure()

# the subsequent entries p(A|A), p(B|B) (the diagonal entries in the coarse-grained Markov matrix)
plt.scatter(p_A_A, p_B_B);
plt.xlabel('p(A|A)')
plt.ylabel('p(B|B)')

# the diagonal line slope 1
plt.plot([0,1], [0,1]);

# the fixed point 1-dimensional manifold
plt.plot([0,1], [1,0]);


# set the transition probability
epsilon = np.random.rand()

# for display purposes, keep it to 2 decimal points
epsilon = round(epsilon,2)

G = nx.DiGraph()

G.add_edge("A", "B", label = "1 - ϵ", weight = 1 - epsilon)
G.add_edge("C", "D", label = "1 - ϵ", weight = 1 - epsilon)
G.add_edge("A", "D", label = 'ϵ', weight = epsilon)
G.add_edge("D", "A", label = '1', weight = 1)
G.add_edge("C", "B", label = "ϵ", weight = epsilon)
G.add_edge("B", "C", label = "1", weight = 1)

# draw networkx graph with graphviz
draw(G, layout='circo')
mat = nx.adj_matrix(G)
markov = mat.toarray()
show_latex_matrix(markov, pre_latex = "A=")
num_trials = 10

compute_markov_states(markov, mapping, num_trials)
# set the transition probability
epsilon = np.random.rand()

# for display purposes, keep it to 2 decimal points
epsilon = round(epsilon,2)

G = nx.DiGraph()

G.add_edge("A", "A", label = "ϵ", weight = epsilon)
G.add_edge("C", "C", label = "ϵ", weight = epsilon)
G.add_edge("A", "C", label = "1 - ϵ", weight = 1 - epsilon)
G.add_edge("C", "A", label = "1 - ϵ", weight = 1 - epsilon)

G.add_edge("D", "D", label = "ϵ", weight = epsilon)
G.add_edge("B", "B", label = "ϵ", weight = epsilon)
G.add_edge("D", "B", label = "1 - ϵ", weight = 1 - epsilon)
G.add_edge("B", "D", label = "1 - ϵ", weight = 1 - epsilon)

# draw networkx graph with graphviz
draw(G, layout='dot')
A_2 = coarse_grain_markov(markov, 2)
show_latex_matrix(A_2, pre_latex = "A^2 =")
A_2
G=nx.from_numpy_matrix(A_2, create_using=nx.DiGraph)
vals = nx.get_edge_attributes(G, 'weight')

nx.set_edge_attributes(G, name = 'label', values = vals)
draw(G, layout='dot')
G = nx.DiGraph()

G.add_edge("A", "B", label = "1-ϵ", weight = epsilon)
G.add_edge("B", "A", label = "1-ϵ", weight = epsilon)

draw(G, layout='dot')
