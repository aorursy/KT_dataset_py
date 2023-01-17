import os

import numpy as np



with open('../input/algo-inputs/test-input-1.txt', 'r') as file:

    redader=file.readlines()

    list_0=[row.rstrip().split() for row in redader]

    city_arr=np.array(list_0).astype('int')

    



pos_cities=city_arr[:,1:]



distances_array = np.array([[np.linalg.norm(pos_cities[i] - pos_cities[j])

                                    for i in range(len(pos_cities))]

                                    for j in range(len(pos_cities))])



def nearest(last,unvisited,edges):

    near=unvisited[0]

    min_dist=edges[last,near]

    for i in unvisited[1:]:

        if edges[last,i]<min_dist:

            near=i

            min_dist=edges[last,near]

    return near



def nearest_neighbour(nodelist,i,edges):

    unvisited=list(range(len(nodelist)))

    unvisited.remove(i)

    last=i

    tour=[i]

    while unvisited!=[]:

        next=nearest(last,unvisited,edges)

        tour.append(next)

        unvisited.remove(next)

        last=next

    return tour



def length(tour,edges):

    tour_length=edges[tour[-1],tour[0]]

    for i in range(1,len(tour)):

        tour_length+=edges[tour[i],tour[i-1]]

    return tour_length





def plot_graph(nodelist,visited):

    for i,txt in enumerate(nodelist):

        plt.scatter(nodelist[i][0],nodelist[i][1])

        #plt.annotate(txt,(nodelist[i][0],nodelist[i][1]))

    for i in range(0,len(visited)):

        plt.plot((nodelist[visited[i][0]][0],nodelist[visited[i][1]][0]),(nodelist[visited[i][0]][1],nodelist[visited[i][1]][1]))

    plt.show()



def graph_mst(opt_tour,nodelist):

    x_coords=[]

    y_coords=[]

    for i in opt_tour:

        x_coords.append(nodelist[i][0])

        y_coords.append(nodelist[i][1])

    for i,txt in enumerate(opt_tour):

        plt.plot(x_coords,y_coords)

        plt.scatter(x_coords,y_coords)

        #plt.annotate(txt,(x_coords[i],y_coords[i]))

    plt.show()



    

def results(tour_length,opt_tour,nodelist,t1):

    print(f'The optimal tour is {opt_tour}')

    print('-----')

    print(f"The length of the tour is {tour_length}")

    print('-----')

    t2=datetime.datetime.now()

    print(f'The time take is {t2-t1}')



    graph_mst(opt_tour,nodelist)

    
import datetime

import matplotlib.pyplot as plt



t1=datetime.datetime.now()

lengths=[]

tours=[]

for i in range(0,len(distances_array)):

    tour=nearest_neighbour(pos_cities,i,distances_array)

    tour_length = length(tour, distances_array)

    tours.append(tour)

    lengths.append(tour_length)

idx=lengths.index(min(lengths))

tour_length=min(lengths)

opt_tour=tours[idx]

#plot_graph(nodelist,visited)

results(tour_length,opt_tour,pos_cities,t1)

file1 = open("output-1.txt","a")

file1.write(str(tour_length)+"\n")

file1.writelines([str(i)+"\n" for i in opt_tour])

file1.close()