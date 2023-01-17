from Levenshtein import distance
import numpy as np


def get_distance_matrix(str_list):
    """ Construct a levenshtein distance matrix for a list of strings"""
    dist_matrix = np.zeros(shape=(len(str_list), len(str_list)))

    print ("Starting to build distance matrix. This will iterate from 0 till ", len(str_list) )
    for i in range(0, len(str_list)):
        print (i)
        for j in range(i+1, len(str_list)):
                dist_matrix[i][j] = distance(str_list[i], str_list[j]) 
    for i in range(0, len(str_list)):
        for j in range(0, len(str_list)):
            if i == j:
                dist_matrix[i][j] = 0 
            elif i > j:
                dist_matrix[i][j] = dist_matrix[j][i]

    return dist_matrix


str_list = [
    "part", "spartan"
  
]
get_distance_matrix(str_list)