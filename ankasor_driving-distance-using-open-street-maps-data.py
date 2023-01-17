"""

Created on Thu Jul 20 20:36:21 2017



Kaggle Kernel to show calculation of driving distances for the NYC Taxi Competition



@author: Friedrich Duge aka Ankasor

"""



import osmnx as ox

import networkx as nx

import os



#Settings for Streetnetwork-Download

STREETGRAPH_FILENAME = 'streetnetwork.graphml'

FORCE_CREATE = False



#This Checks if the Streetnetwork File exists (or creation is overwritten using FORCE_CREATE)

if (not os.path.isfile(".//data//"+STREETGRAPH_FILENAME) )or FORCE_CREATE:

    #There are many different ways to create the Network Graph. See the osmnx documentation for details

    area_graph = ox.graph_from_place('New York, USA', network_type='drive_service')

    ox.save_graphml(area_graph, filename=STREETGRAPH_FILENAME)

else:

    area_graph = ox.load_graphml(STREETGRAPH_FILENAME)



def driving_distance(area_graph, startpoint, endpoint):

    """

    Calculates the driving distance along an osmnx street network between two coordinate-points.

    The Driving distance is calculated from the closest nodes to the coordinate points.

    This can lead to problems if the coordinates fall outside the area encompassed by the network.

    

    Arguments:

    area_graph -- An osmnx street network

    startpoint -- The Starting point as coordinate Tuple

    endpoint -- The Ending point as coordinate Tuple

    """

    

    #Find nodes closest to the specified Coordinates

    node_start = ox.utils.get_nearest_node(area_graph, startpoint)

    node_stop = ox.utils.get_nearest_node(area_graph, endpoint)

    

    #Calculate the shortest network distance between the nodes via the edges "length" attribute

    distance = nx.shortest_path_length(area_graph, node_start, node_stop, weight="length")

    

    return distance



#Usage example:

#Coordinates of the first drive in the train dataset

startpoint = (40.767937, -73.982155)

endpoint = (40.765602, -73.964630)



print (driving_distance(area_graph, startpoint, endpoint))

#Output (driving distance in meters): 1955.27332523963