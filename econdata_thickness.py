# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import Image
Image(filename='../input/deepspeacai/Optimal Layout - GANs Testing/21.PNG',width=1200, height=300)

!pip install sknw
import pandas as pd

import cairo

import matplotlib.pylab as plt

import math

from IPython.display import Image

import numpy as np

from numpy import *

import glob

import os

import os.path

import time

import cv2

import random

import ast

from PIL import Image

from math import *

import networkx as nx

import matplotlib.cm as cm

from matplotlib.pyplot import figure, show, rc

from scipy.ndimage.interpolation import geometric_transform

from skimage.morphology import skeletonize

from skimage import data

import sknw

import random

from shapely.geometry import LineString

def get_skeleton(img_2,print_):

    # open and skeletonize

    img = np.abs(np.round(img_2[:,:,0]/255).astype(np.int)-1)

    img_white = np.round(img_2[:,:,0]/255).astype(np.int)

    ske = skeletonize(img).astype(np.uint16)



    # build graph from skeleton

    graph = sknw.build_sknw(ske)



    if(print_==True):

        plt.figure(figsize=(10,10))

        # plt.imshow(img_white, cmap='gray')



    # draw edges by pts

    if(print_==True):

        for (s,e) in graph.edges():

            ps = graph[s][e]['pts']

            plt.plot(ps[:,1], ps[:,0], 'black')





    # draw node by o

    node, nodes = graph.node, graph.nodes()





    if(print_==True):

        plt.title('Skeleton')

        plt.axis("off")

        plt.show()



    return graph, img_white



def get_len(x1,y1,x2,y2):

    length = math.sqrt((x2-x1)**2+(y2-y1)**2)

    return length
def createLineIterator(P1, P2, img):

    """

    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points



    Parameters:

        -P1: a numpy array that consists of the coordinate of the first point (x,y)

        -P2: a numpy array that consists of the coordinate of the second point (x,y)

        -img: the image being processed



    Returns:

        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])

    """

    #define local variables for readability

    imageH = img.shape[0]

    imageW = img.shape[1]

    P1X = P1[0]

    P1Y = P1[1]

    P2X = P2[0]

    P2Y = P2[1]



    #difference and absolute difference between points

    #used to calculate slope and relative location between points

    dX = P2X - P1X

    dY = P2Y - P1Y

    dXa = np.abs(dX)

    dYa = np.abs(dY)



    #predefine numpy array for output based on distance between points

    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)

    itbuffer.fill(np.nan)



    #Obtain coordinates along the line using a form of Bresenham's algorithm

    negY = P1Y > P2Y

    negX = P1X > P2X

    if P1X == P2X: #vertical line segment

       itbuffer[:,0] = P1X

       if negY:

           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)

       else:

           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)

    elif P1Y == P2Y: #horizontal line segment

       itbuffer[:,1] = P1Y

       if negX:

           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)

       else:

           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)

    else: #diagonal line segment

       steepSlope = dYa > dXa

       if steepSlope:

           slope = dX.astype(np.float32)/dY.astype(np.float32)

           if negY:

               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)

           else:

               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)

           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X

       else:

           slope = dY.astype(np.float32)/dX.astype(np.float32)

           if negX:

               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)

           else:

               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)

           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y



    #Remove points outside of image

    colX = itbuffer[:,0]

    colY = itbuffer[:,1]

    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]



    #Get intensities from img ndarray

    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]



    return itbuffer

def wall_analysis(img_path,edge_id):



    k=edge_id



    img_h69 = cv2.imread(img_path)

    graph_h69, image_h69 = get_skeleton(img_h69,print_=False)



    w,h,d = img_h69.shape



    edges_list_h69 = [graph_h69[s][e]['pts'] for (s,e) in graph_h69.edges()]



    #plt.figure(figsize=(10,10))

    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(131)

    ax1.axis("off")



    # draw edges by pts

    for (s,e) in graph_h69.edges():

        ps = graph_h69[s][e]['pts']

        ax1.plot(ps[:,1], ps[:,0], 'red')



    ######SET PARAMETERS#########

    #############################

    #############################



    #Select edge

    #k = 2

    #number of point to sample

    nb_point = 30

    #radius

    radius= 40



    #############################

    #############################

    #############################



    index_ = np.linspace(0,len(edges_list_h69[k])-1,num=nb_point).astype(int)



    middle_id = int(len(edges_list_h69[k])/2)

    middle_before = middle_id-1

    middle_before = middle_id+1



    x_pt = edges_list_h69[k][middle_id][1]

    y_pt = edges_list_h69[k][middle_id][0]



    padding = 100



    pts= [edges_list_h69[k][j] for j in index_]



    wall_thickness = []

    for i in range(1,len(pts)-1):



        v= pts[i+1]-pts[i-1]

        n = [v[1],-v[0]]



        u = n/np.linalg.norm(n)



        new_pt_1 = pts[i]-radius*u

        new_pt_2 = pts[i]+radius*u



        pixel_values = createLineIterator([new_pt_1[1].astype(int),new_pt_1[0].astype(int)], [new_pt_2[1].astype(int),new_pt_2[0].astype(int)], image_h69).astype(int)

        wall_thick = np.sum(pixel_values[:,2]!=1)



        wall_thickness.append(wall_thick)



        ax1.plot([new_pt_1[1],new_pt_2[1]],[new_pt_1[0],new_pt_2[0]],"c")

        ax1.plot(new_pt_2[1],new_pt_2[0],"r.")

        ax1.plot(new_pt_1[1],new_pt_1[0],"g.")



    ax1.imshow(image_h69, cmap='gray')



    ax2 = fig.add_subplot(132)

    ax2.plot(wall_thickness,"black")

    ax2.set_title("Thickness")

    ax2.set_xlabel("Wall Linear")

    ax2.set_ylabel("Wall Thickness (in pixels)")

    ax2.set_ylim([0,np.max(wall_thickness)+4])

    ax2.set_xlim([0,len(wall_thickness)-1])

    plt.grid()



    delta_thick = []

    for i in range(1,len(wall_thickness)):

        delta = (wall_thickness[i]-wall_thickness[i-1])

        delta_thick.append(delta)



    ax3 = fig.add_subplot(133)

    ax3.plot(delta_thick,"black")

    ax3.set_title("Texture")

    ax3.plot([len(wall_thickness)-3,0],[0,0],"r")

    ax3.set_xlabel("Wall Linear")

    ax3.set_ylabel("Wall Texture (in pixels)")

    ax3.set_ylim([-5,5])

    ax3.set_xlim([0,len(wall_thickness)-3])

    plt.grid()

img_path = "../input/deepspeacai/Optimal Layout - GANs Testing/73.PNG"

wall_analysis(img_path,edge_id=2)

img_names = os.listdir("../input/deepspeacai/Optimal Layout - GANs Testing")

img_path = "IMG/crazy_plans/clean/"+img_names[0]