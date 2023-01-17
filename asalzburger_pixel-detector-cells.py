import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackml
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
# One event of 8850
event_id = 'event000001000'
# "All methods either take or return pandas.DataFrame objects"
hits, cells, particles, truth = load_event('../input/train_1/'+event_id)
h_id = 19144 
pixel_cluster = cells[ cells['hit_id']==h_id ]
len(pixel_cluster)
# a function that calculates the cluster size and makes a pixel matrix
def pixel_matrix(pixel_cluster, show=False):
    # cluster size
    min0 = min(pixel_cluster['ch0'])
    max0 = max(pixel_cluster['ch0'])
    min1 = min(pixel_cluster['ch1'])
    max1 = max(pixel_cluster['ch1'])
    # the matrix
    matrix = np.zeros(((max1-min1+3),(max0-min0+3)))
    for pixel in pixel_cluster.values :
        i0 = int(pixel[1]-min0+1)
        i1 = int(pixel[2]-min1+1)
        value = pixel[3]
        matrix[i1][i0] = value 
    # return the matris
    if show :
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.YlOrRd)
        plt.colorbar()
        plt.show()
    return matrix, max0-min0+1, max1-min1+1
cluster,width,length = pixel_matrix(pixel_cluster,True)
print(width,length)
pixel_hit = hits[ hits['hit_id']==h_id ]
print(pixel_hit)
truth_pixel_hit = truth[ truth['hit_id']==h_id]
print(truth_pixel_hit)
detector = pd.read_csv('./../input/detectors.csv')
# method to retrieve the according module associated to a hit
def retrieve_module(detector,hit) :
    volume = detector[ detector['volume_id']==hit.volume_id.data[0] ]
    layer  = volume[ volume['layer_id']==hit.layer_id.data[0] ]
    module = layer[ layer['module_id']== hit.module_id.data[0] ]
    return module
# get the one for our example
module = retrieve_module(detector,pixel_hit)
# method to build and nomralize direction vector from the 
def direction_vector(ipx, ipy, ipz) :
    # the absolute momentum for normalization
    p = np.sqrt(ipx*ipx+ipy*ipy+ipz*ipz)
    # build the direction vector - to be used with the matrix 
    direction = [[ipx/p], [ipy/p], [ipz/p]]
    return direction
# get the truth direction at the module, it's more accurate than the starting position
direction_global_hit = direction_vector(truth_pixel_hit.tpx.data[0],truth_pixel_hit.tpy.data[0],truth_pixel_hit.tpz.data[0])
print(direction_global_hit)
# get the truth particle information
particle = particles[ particles['particle_id'] == truth_pixel_hit.particle_id.data[0] ]
print(particle)
# build the direction vector with the start momentum
direction_global_start = direction_vector(particle.px.data[0],particle.py.data[0],particle.pz.data[0])
print(direction_global_start)
# extract phi and theta from a direciton vector
def phi_theta(dx,dy,dz) :
    dr  = np.sqrt(dx*dx+dy*dy)
    phi = np.arctan2(dy,dx)
    theta = np.arctan2(dr,dz)
    return phi, theta
# get thet and phi
phi_hit, theta_hit = phi_theta(direction_global_hit[0][0],
                               direction_global_hit[1][0],
                               direction_global_hit[2][0])
print(theta_hit)
# get the length of the cluster in v direction
cluster_length_v_hit = np.abs(2.*module.module_t.data[0]/np.tan(theta_hit))
cluster_size_v_hit   = cluster_length_v_hit/module.pitch_v.data[0]
print(cluster_length_v_hit,cluster_size_v_hit)
# function to extract the rotation matrix (and its inverse) from module dataframe
def extract_rotation_matrix(module) :
    rot_matrix = np.matrix( [[ module.rot_xu.data[0], module.rot_xv.data[0], module.rot_xw.data[0]],
                            [  module.rot_yu.data[0], module.rot_yv.data[0], module.rot_yw.data[0]],
                            [  module.rot_zu.data[0], module.rot_zv.data[0], module.rot_zw.data[0]]])
    return rot_matrix, np.linalg.inv(rot_matrix)
module_matrix, module_matrix_inv = extract_rotation_matrix(module)
print (module_matrix)
direction_local_hit =  module_matrix_inv*direction_global_hit
print(direction_local_hit)
# theta is defined as the arctan of the radial vs the longitudinal components
# phi is defined as the acran of the two transvese components
phi_local,theta_local = phi_theta(direction_local_hit[0][0],
                                  direction_local_hit[1][0],
                                  direction_local_hit[2][0])
print(phi_local,theta_local)
path_in_silicon = 2*module.module_t.data[0]/np.cos(theta_local)
print(path_in_silicon)
# calculate the component in u and v
path_component_u = path_in_silicon*np.sin(theta_local)*np.cos(phi_local)
path_component_v = path_in_silicon*np.sin(theta_local)*np.sin(phi_local)
cluster_size_in_u = path_component_u/module.pitch_u.data[0]
cluster_size_in_v = path_component_v/module.pitch_v.data[0]
# print the cluster size 
print(cluster_size_in_u, cluster_size_in_v)
