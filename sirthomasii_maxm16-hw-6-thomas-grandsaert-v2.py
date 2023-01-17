import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from skimage.io import imread, imsave
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # a nice progress bar
import pandas as pd
stack_image = imread('../input/rec_8bit_ph03_cropC_kmeans_scale510.tif')
from skimage.morphology import binary_opening, convex_hull_image as chull
bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])
bubble_inver=np.invert(bubble_image)
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt as distmap
bubble_dist = distmap(bubble_inver)
from skimage.feature import peak_local_max
bubble_candidates = peak_local_max(bubble_dist, min_distance=12)
from skimage.morphology import watershed
bubble_seeds = peak_local_max(bubble_dist, min_distance=12, indices=False)
markers = ndi.label(bubble_seeds)[0]
from skimage.measure import regionprops
import sys
from io import StringIO
from IPython import get_ipython
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from scipy.spatial import ConvexHull # to compute area from the given points
from mpl_toolkits.mplot3d import Axes3D
import csv 
import math
from scipy.ndimage import interpolation
import skimage
import timeit
from skimage import segmentation

class IpyExit(SystemExit):
    def __init__(self):
        # print("exiting")  # optionally print some message to stdout, too
        # ... or do other stuff before exit
        sys.stderr = StringIO()

    def __del__(self):
        sys.stderr.close()
        sys.stderr = sys.__stderr__  # restore from backup
def ipy_exit():
    raise IpyExit
if get_ipython():    # ...run with IPython
    exit = ipy_exit  # rebind to custom exit
else:
    exit = exit      # just make exit importable
x_bins=100
y_bins=300
z_bins=300
y_offset=100
z_offset=100

cropped_markers = markers[200:(200+x_bins),y_offset:(y_offset+y_bins),z_offset:(z_offset+z_bins)]
cropped_bubble_dist=bubble_dist[200:(200+x_bins),y_offset:(y_offset+y_bins),z_offset:(z_offset+z_bins)]
cropped_bubble_inver=bubble_inver[200:(200+x_bins),y_offset:(y_offset+y_bins),z_offset:(z_offset+z_bins)]
labeled_bubbles= watershed(-cropped_bubble_dist, cropped_markers, mask=cropped_bubble_inver)
regions=regionprops(labeled_bubbles)
start = timeit.default_timer()

waterVoxels = np.asarray(np.nonzero((stack_image.flatten() == 255)))
airVoxels = np.asarray(np.nonzero((stack_image.flatten() == 0)))
waterVoxels[waterVoxels > 0] = 1
airVoxels[airVoxels > 0] = 1 

print(np.sum(waterVoxels)/np.sum(airVoxels))

stop = timeit.default_timer()
print('Time: ', stop - start)  

start = timeit.default_timer()

avg_size = 0;
bubble_dict = {} #build a dictionary for our bubblely friends

#---------FINDING SCIKIT PROPERTIES THAT ARE SUPPORTED FOR 3D OBJECTS 
#--------(COORDINATES/SIZE/CENTROIDS/SURFACE AREA[THROUGH CONVEXHULL])

for bubble_num in range(len(regions)):
    bubble_dict[bubble_num] = {}
    bubble_dict[bubble_num]["coords"] = regions[bubble_num].coords;
    bubble_dict[bubble_num]["size"] = regions[bubble_num].equivalent_diameter;
    bubble_dict[bubble_num]["centroid"] = regions[bubble_num].centroid;
    bubble_dict[bubble_num]["surface_area"] = ConvexHull(np.asarray(regions[bubble_num].coords)).area
    avg_size += regions[bubble_num].equivalent_diameter;

avg_size = avg_size/len(regions)
num_bubbles = len(regions)

stop = timeit.default_timer()
print('Time: ', stop - start)  
start = timeit.default_timer()

for bubble_num in range(len(regions)):
    
    x_index = int(bubble_dict[bubble_num]["centroid"][0])
    regions_slice=regionprops(labeled_bubbles[x_index,:,:])

    for bubble_num_slice in range(len(regions_slice)):
    
        for index, coord in enumerate(regions_slice[bubble_num_slice].coords):
            y_0 = coord[0]
            z_0 = coord[1]
            y = bubble_dict[bubble_num]["centroid"][1]
            z = bubble_dict[bubble_num]["centroid"][2]
            thresh = 5e-1
            if (abs(y_0 - y)<thresh) and (abs(z_0- z)<thresh):
                bubble_dict[bubble_num]["orientation"] = regions_slice[bubble_num_slice].orientation;
                bubble_dict[bubble_num]["eccentricity"] = regions_slice[bubble_num_slice].eccentricity;
                bubble_dict[bubble_num]["major_axis"] = regions_slice[bubble_num_slice].major_axis_length;
                bubble_dict[bubble_num]["minor_axis"] = regions_slice[bubble_num_slice].minor_axis_length;

stop = timeit.default_timer()
print('Time: ', stop - start)  
start = timeit.default_timer()

ax = a3.Axes3D(plt.figure())
ax.dist=10
ax.azim=40
ax.elev=10
ax.set_xlim3d(0,x_bins)
ax.set_ylim3d(0,y_bins)
ax.set_zlim3d(0,z_bins)

for bubble_num in range(len(regions)):
#     print(bubble_num, end='\r', flush=True)

    pts = np.array(bubble_dict[bubble_num]["coords"])
    hull = ConvexHull(pts)
    faces = hull.simplices
    c=colors.rgb2hex(sp.rand(3))
    c=np.random.rand(3,)
    faces = hull.simplices
    for s in faces:
        sq = [
            [pts[s[0], 0], pts[s[0], 1], pts[s[0], 2]],
            [pts[s[1], 0], pts[s[1], 1], pts[s[1], 2]],
            [pts[s[2], 0], pts[s[2], 1], pts[s[2], 2]],
        ]        
        f = a3.art3d.Poly3DCollection([sq])
        f.set_color(c)
        f.set_edgecolor('k')
        f.set_alpha(0.1)
        ax.add_collection3d(f)
        
plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start)  
start = timeit.default_timer()

boundaries = skimage.segmentation.find_boundaries(labeled_bubbles, mode='thick')
boundaries = np.nonzero(boundaries)
boundaries = np.asarray(list(zip(boundaries[0],boundaries[1],boundaries[2])))

#--------DRAW A BOX AROUND EACH BUBBLE AND LOAD THE BOUNDARIES COORDINATES WHICH ARE WITHIN THE BOX,
#--------THEN LOAD THE BOUNDARY POINTS INTO THE DICTIONARY FOR EACH RESPECTIVE BUBBLE (2PX RADIUS)-----

for bubble_num in range(len(regions)):
    
    bubble_dict[bubble_num]["boundaries"]=[]
    max_x = np.max(regions[bubble_num].coords,axis=0)[0]
    min_x = np.min(regions[bubble_num].coords,axis=0)[0]
    max_y = np.max(regions[bubble_num].coords,axis=0)[1]
    min_y = np.min(regions[bubble_num].coords,axis=0)[1]
    max_z = np.max(regions[bubble_num].coords,axis=0)[2]
    min_z = np.min(regions[bubble_num].coords,axis=0)[2]

    bubble_boundaries_indices = np.nonzero((boundaries[:,0]<max_x) & (boundaries[:,0]>min_x) & (boundaries[:,1]<max_y) & (boundaries[:,1]>min_y) & (boundaries[:,2]<max_z) & (boundaries[:,2]>min_z))
    bubble_dict[bubble_num]["boundaries"] = bubble_boundaries_indices[0]

#-------DETERMINE THE COORDINATES WHICH ARE SHARED BETWEEN DIFFERENT BUBBLES AND ADD AN ASSOCIATION 
for bubble_num_i in range(len(regions)):
    
    if "num_neighbors" not in bubble_dict[bubble_num_i].keys():
            bubble_dict[bubble_num_i]["num_neighbors"] = 0
    if "neighbors" not in bubble_dict[bubble_num_i].keys():
            bubble_dict[bubble_num_i]["neighbors"] = []
            
    for bubble_num_j in range(len(regions)):
        
        if "num_neighbors" not in bubble_dict[bubble_num_j].keys():
                bubble_dict[bubble_num_j]["num_neighbors"] = 0
        if "neighbors" not in bubble_dict[bubble_num_j].keys():
                bubble_dict[bubble_num_j]["neighbors"] = []
        
        if((bubble_num_i != bubble_num_j) and (bubble_num_j not in bubble_dict[bubble_num_i]["neighbors"])):
           
            if (any(x in bubble_dict[bubble_num_i]["boundaries"] for x in bubble_dict[bubble_num_j]["boundaries"])): 
                
                    bubble_dict[bubble_num_i]["num_neighbors"] += 1
                    bubble_dict[bubble_num_j]["num_neighbors"] += 1
                    bubble_dict[bubble_num_i]["neighbors"].append(bubble_num_j)
                    bubble_dict[bubble_num_j]["neighbors"].append(bubble_num_i)
                    
stop = timeit.default_timer()
print('Time: ', stop - start)  
with open('bubble_props.csv', 'w') as f:
    for bubble_num in range(len(regions)):
        f.write("%s,%s\n"%("BUBBLE NUMBER",bubble_num))
        for key in bubble_dict[bubble_num].keys():
            if key != "coords":
                f.write("%s,%s\n"%(key,bubble_dict[bubble_num][key]))
start = timeit.default_timer()

#--------(ORIENTATION/MAJOR/MINOR/SPHERICITY)
#-----------BUILDING DIMENSIONS OF THE PROBING IDENTITY MATRIX FOR YZ PLANE------------
x_id_span=100
y_id_span=100
yz_ID = np.identity(y_id_span)
angle_step = 5 

for bubble_num in range(len(regions)):

    angle = 0
    max_sum = 0
    max_sum_angle = 0

    #----------CENTROID COORDS---------------
    x_c = int(bubble_dict[bubble_num]["centroid"][0])
    y_c = int(bubble_dict[bubble_num]["centroid"][1])
    z_c = int(bubble_dict[bubble_num]["centroid"][2])
    
    for angle_i in range(int(360/angle_step)):
        angle += 5
        
        #--------------BUILD THE ZERO MATRIX BASED ON THE X/Y/Z BIN NUMBERS
        
        xyz_A=[] #MATRIX WITH THE CURRENT BUBBLE REGION INSERTED
        xyz_B=[] #3D IDENTITY MATRIX ROTATED AT A ARBITRARY ANGLE AND TRANSLATED TO THE CURRENT BUBBLE CENTROID
        yz_A = np.zeros((y_bins,z_bins))
        yz_B = np.zeros((y_bins,z_bins))       

        #----------ROTATE THE IDENTITY MATRIX, CONVERT ALL NONZEROS TO 1-----------------
        rotated_ID = skimage.transform.rotate(yz_ID, angle, resize=False, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
        rotated_ID[rotated_ID > 0] = 1

        #-----------MOVE THE ROTATED IDENTITY MATRIX TO THE CENTROID OF THE BUBBLE, THEN PROJECT IT IN 3D SPACE

        for index, x in np.ndenumerate(rotated_ID):
            
            if (x==1) and (index[0]+y_c-int(y_id_span/2)<300) and (index[1]+z_c-int(y_id_span/2)<300):
                yz_B[index[0]+y_c-int(y_id_span/2)][index[1]+z_c-int(y_id_span/2)] = 1
                
            elif (index[0]+y_c-int(y_id_span/2)<300) and (index[1]+z_c-int(y_id_span/2)<300):
                yz_B[index[0]+y_c-int(y_id_span/2)][index[1]+z_c-int(y_id_span/2)] = 0
                
        for j in range(x_id_span):
            xyz_B.append(yz_B)
        
        #--------------ADD CURRENT BUBBLE REGION TO ZERO VALUE 3D MATRIX--------------------
        for j in range(x_bins):
            xyz_A.append(yz_A)

        for index, coord in enumerate(regions[bubble_num].coords):
            xyz_A[coord[0]][coord[1]][coord[2]] = 1
            
        #-------MULTIPLY THE ROTATED 3D MATRIX WITH THE BUBBLE REGION MATRIX ELEMENT BY ELEMENT AND SUM THE PRODUCT
        sum_region = np.sum(np.multiply(xyz_A,xyz_B))

        if (sum_region > max_sum):
            max_sum = sum_region
            max_sum_angle = angle
            max_sum_matrix = xyz_B
            bubble_dict[bubble_num]["orientation_xy"] = angle
     
    #-------MAXIMUM ANGLE IS FOUND FOR XY, COMPUTE THE MAJOR AXIS LENGTH FROM MAX/MIN X INDICES------
    matrix_prod_indices = np.multiply(xyz_A[x_c],max_sum_matrix[x_c]).nonzero()
    pt1_x = np.amax(matrix_prod_indices[0])
    pt1_x_i = (matrix_prod_indices[0].argmax(axis=0))
    
    pt2_x = np.amin(matrix_prod_indices[0])
    pt2_x_i = (matrix_prod_indices[0].argmin(axis=0))
    
    pt1_y = matrix_prod_indices[1][pt1_x_i]
    pt2_y = matrix_prod_indices[1][pt2_x_i]

    bubble_dict[bubble_num]["major_axis_xy"] = math.sqrt((pt2_x - pt1_x)**2 + (pt2_y - pt1_y)**2)  

    #----------ROTATE THE MAX IDENTITY MATRIX 90 DEGREES TO FIND MINOR AXIS LENGTH-----------------
    max_sum_matrix_rot90 = skimage.transform.rotate(yz_ID, max_sum_angle+90, resize=False, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
    max_sum_matrix_rot90[max_sum_matrix_rot90 > 0] = 1
    max_sum_matrix_rot90_temp = np.zeros((y_bins,z_bins))
    
    for index, x in np.ndenumerate(max_sum_matrix_rot90):
        if (x==1) and (index[0]+y_c-int(y_id_span/2)<300) and (index[1]+z_c-int(y_id_span/2)<300):
            max_sum_matrix_rot90_temp[index[0]+y_c-int(y_id_span/2)][index[1]+z_c-int(y_id_span/2)] = 1
            
        elif (index[0]+y_c-int(y_id_span/2)<300) and (index[1]+z_c-int(y_id_span/2)<300):
            
            max_sum_matrix_rot90_temp[index[0]+y_c-int(y_id_span/2)][index[1]+z_c-int(y_id_span/2)] = 0
            
    max_sum_matrix_rot90 = max_sum_matrix_rot90_temp
    
    #-------COMPUTE THE MINOR AXIS LENGTH FROM MAX/MIN X INDICES------
    matrix_prod_indices = np.multiply(xyz_A[x_c],max_sum_matrix_rot90).nonzero()
    
    pt1_x = np.amax(matrix_prod_indices[0])
    pt1_x_i = (matrix_prod_indices[0].argmax(axis=0))
    
    pt2_x = np.amin(matrix_prod_indices[0])
    pt2_x_i = (matrix_prod_indices[0].argmin(axis=0))
    
    pt1_y = matrix_prod_indices[1][pt1_x_i]
    pt2_y = matrix_prod_indices[1][pt2_x_i]

    bubble_dict[bubble_num]["minor_axis_xy"] = math.sqrt((pt2_x - pt1_x)**2 + (pt2_y - pt1_y)**2) 
    bubble_dict[bubble_num]["eccentricity_xy"] = bubble_dict[bubble_num]["minor_axis_xy"]/bubble_dict[bubble_num]["major_axis_xy"]
    
fig = plt.figure(figsize=(10, 5))
a=fig.add_subplot(2,3,1)
imgplot = plt.imshow(max_sum_matrix[int(bubble_dict[bubble_num]["centroid"][0])])

a=fig.add_subplot(2,3,2)
imgplot = plt.imshow(xyz_A[int(bubble_dict[bubble_num]["centroid"][0])])

a=fig.add_subplot(2,3,3)
imgplot2 = plt.imshow(np.multiply(xyz_A[x_c],max_sum_matrix[x_c]))

a=fig.add_subplot(2,3,4)
imgplot = plt.imshow(max_sum_matrix_rot90)

a=fig.add_subplot(2,3,5)
imgplot = plt.imshow(xyz_A[int(bubble_dict[bubble_num]["centroid"][0])])

a=fig.add_subplot(2,3,6)
imgplot2 = plt.imshow(np.multiply(xyz_A[x_c],max_sum_matrix_rot90))

plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start)  