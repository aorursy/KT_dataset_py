import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import matplotlib.pyplot as pplt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# We shall use matplotlibe to draw the point on 2D plane. In the future kernels, i would also show how to use Seaborn,Bokeh etc

x=[3,4,6,7,1,2,8] #numpy array to store x values

y=[1,2,3,8,3,5,3] #numpy array to store y values

point =[5,4]

pplt.scatter(x,y,s=200,c='blue')

pplt.scatter(point[0],point[1],s=200,c='green')
#Problem Statement: Find the 1 nearest neighbor of a given point

# If you remember the basics of programming, we use functions or procedures to develop programs, the advantages include reusability, modularity, 

# ease of management etc.



#pdataset contains the points in which, we shall identify the nearest neighbor

#point contains the point and k is the number of neighbor returned back 

# In order to keep things simple, i am using sequential approach, like reading the points dataset in sequential order. 

# Later, we shall see, how to improve this algorithm further to overcome some of the challenges/limitations of K Neareast Neighbors

def k_nearest_neighbors(pdataset,point,k):    

    #minimum distance will store the minimum euclidean distance of a point in the space

    minimum_dist = 0.0

    # we define the counter to store first distance without any condition as we are following the sequential approach

    counter = 0    

    #npoint will store the closest point/neighbor of the provided point

    npoint=[]    

    #Lets traverse the loop for all points in the dataset

    for pn in pdataset:

        # instead of writing our own function for calculating difference and square roots, we would use numpy libraray to calculate 

        #Euclidean distance of each point in our dataset from the given point

        dist = np.sqrt(np.sum(np.square(pn-point)))        

        #the following condition will assign first point without any checking and we assume the first point has the minimum distance to  

        # the given point

        if counter==0:

            minimum_dist=dist

            npoint=pn

        # However, when we have our first our, then we need to compare the calculaed distance and if the distance is minimum with the existing 

        # value of mimimum_dist, it indicates we have a new result, So we overwrite the minimym_dist and npoint with their new values

        else:

            if dist<minimum_dist:

                npoint=pn

                minimum_dist = dist        

        counter = counter + 1        

    #Once the above loop will finish, it will store the closest point in npoint and store the minimum distance in  minimum_distance

    # and in the end, we shall return the closest neighbor

    print("minimum distance ",minimum_dist)

    return  npoint
# Lets define the dataset using X and Y

#initially, we declare our datasets with 0s. As we have 14 values.So we would initialize our dataset with 14 values and reshape it with (7,2)

# which means we have now 2D array, if you don't understand the shape, arrays and 2D arrays. Don't worry, I would add numpy tutorial shortly to 

# make you expert. At this stage, just remember, following statement will declare the points x and y in an array shape to calculate euclidean

#distance

dataset = (np.zeros(14)).reshape(7,2)

#Assign x values to fist index/dimension

dataset[...,0]=x

#Assign y values to Second dimension

dataset[...,1]=y

# if you want print the following statment to see the values at both index. "..." is used when, we want to display all values at one particular

# index in multi-dimensional array

print(dataset[...,0])

print(dataset[...,1])

# Last to verfiy the shape before callling our K Nearest Neighbor algorithm With K=1

dataset.shape

result_point=k_nearest_neighbors(dataset,point,1)
#Lets plot the dataset,point and the nearest neighbor to varify the algorithm

print(result_point)

pplt.scatter(dataset[...,0],dataset[...,1],s=200,c='blue')

pplt.scatter(point[0],point[1],s=200,c='green')

pplt.scatter(result_point[0],result_point[1],s=200,c='red')
def adjust_dist_weights(npointdic,pnpointindex,pdist,pnpoints,counter,k):         

    npointdic[pdist]=pnpoints      

    pnpointindex[pdist]=counter

    rnpoints={}    

    npointindex={}

    

    count=0

    if counter<k:

        rnpoints=npointdic

        npointindex=pnpointindex

    else:                    

        for key in sorted(npointdic.keys()):

            if count<k:

                rnpoints[key]=npointdic[key]                

                npointindex[key]=pnpointindex[key]

            else:

                break                

                #rnpoints=sorted(npointdic.keys())

            count = count + 1    

    #print(rnpoints)    

    return rnpoints,npointindex
def k_nearest_neighbors(pdataset,point,k):

    old_dist = []

    counter = 0    

    npoint={}

    npointindex={}

#     for i in range(k):

#         npoint[float(i)] = [0.0,0.0]

#         print(i)

    

    for pn in pdataset:

        dist = np.sqrt(np.sum(np.square(pn-point)))        

        npoint,npointindex=adjust_dist_weights(npoint,npointindex,dist,pn,counter,k)        

        counter = counter + 1        

#     print("final results")

#     print(npoint)

    return  npoint,npointindex
mydatasets = (np.random.rand(100)*100).reshape(50,2)

mydatasets.shape
%time

point = [57.23345,64.3456]

k_point,k_point_index = k_nearest_neighbors(mydatasets,point,3)

print(k_point.values())

print(k_point_index.values())
pplt.scatter(mydatasets[:,0],mydatasets[:,1],s=200,c='blue')

pplt.scatter(point[0],point[1],s=200,c='green')

for key in k_point.keys():

    pplt.scatter(k_point[key][0],k_point[key][1],s=200,c='red')