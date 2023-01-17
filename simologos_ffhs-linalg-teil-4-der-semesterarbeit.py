import networkx as nx

import matplotlib.pyplot as plt

import numpy as np
def linkmatrix(g):

    """Returns a matrix similar to the google matrix.

    Similar means, that dangling nodes (nodes with no output link)

    get a default weight assigned which is 1 devided by the amount of nodes

    provided.

    

    Args:

        param g (networkx graph): The graph for which the link matrix should be calculated.



    Returns:

        The link matrix

    """

    matrix = nx.to_numpy_matrix(g, nodelist=list(g), weight='weight')

    

    if len(g) == 0:

        return matrix 



    danglingWeights = np.repeat(1.0 / len(g), len(g))        

    danglingNodes = np.where(matrix.sum(axis=1) == 0)[0]



    # Assign dangling_weights to any dangling nodes

    for node in danglingNodes:

        matrix[node] = danglingWeights



    # Normalize rows to sum to 1

    matrix /= matrix.sum(axis=1)



    return 0.85 * matrix + (1 - 0.85) * danglingWeights



def pagerank(g):

    """ Calculates the pagerank for a given graph by calculating the eigenvectors

    with numpy.

    

    Args:

        param g (networkx graph): The graph for which the link matrix should be calculated.



    Returns:

        The pageranks of all nodes in the graph.

    """

    if len(g) == 0:

        return {}

    

    M = linkmatrix(g)

    

    # use numpy to calculate the eigenvalues and eigenvectors

    eValues, eVectors = np.linalg.eig(M.T)

    ind = np.argmax(eValues)

    

    # eigenvector of largest eigenvalue is at ind, normalized

    largest = np.array(eVectors[:, ind]).flatten().real

    norm = float(largest.sum())

    

    return dict(zip(g, map(float, largest / norm)))
# Generate a graph.

g = nx.gnp_random_graph(7, 0.5, directed=True)

nx.draw(g, with_labels=True)



# Plot the graph.

plt.show()



# Calculate the pagerank.

print(pagerank(g))