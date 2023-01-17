import networkx as nx

import random

import time

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import spline

import csv



def CreateGraph():

  g=nx.Graph()

  file=open('../input/CA-CondMat.txt',newline='\n')

  #creating a graph from the relationship given in the condmat file

  for word in file:

    data=word.split()

    u=data[0]

    v=data[1]

    g.add_edge(u,v)    

  #total no of nodes and edges formed in the network

  g.remove_edges_from(g.selfloop_edges())

  print("Number of Nodes =" ,g.number_of_nodes())  

  print("Number of Edges =" ,g.number_of_edges())    

  return g







def Coreness(g,k_shell):

  coreness={}

  for node in g.nodes():

    sum=0

    for neighbor in g.neighbors(node):

        sum=sum + k_shell[neighbor]   

    #sum=sum+d[node]

    coreness[node]=sum

 

  return coreness

 



def Extended_Coreness(g,coreness):

  extended_coreness={}

  for node in g.nodes():

    sum=0

    for neighbor in g.neighbors(node):

        sum=sum+coreness[neighbor]

    extended_coreness[node]=sum

  

  return extended_coreness

        



def Eigen_Vector_Centrality(g):  

  eigen_vector_centrality=nx.eigenvector_centrality(g)

  return eigen_vector_centrality



def H_Index(g):

    h_index={}

    for node in g.nodes():

       max_h_index=g.degree(node) 

       result=-1

       for i in range (0,max_h_index+1):

           total_nodes=0

           for neighbor in g.neighbors(node):

               if(g.degree(neighbor)>=i):

                   total_nodes=1+total_nodes;

           if(total_nodes>=i):

               result=max(result,i)

       h_index[node]=result      

    return h_index



def Cluster_Coefficient(g):

    cluster_rank=nx.clustering(g)

    return cluster_rank



def SIRModel(g,spreaders,beta,simulations):

# infected_scale=np.array(np.zeros(50))

 ftc=0

 while simulations>0 :

   simulations=simulations-1

   #print(simulations)

   infected=spreaders

    

   status={}

   for i in g.nodes():

      status[i]=0;

   for i in infected:

     status[i]=1;

  

   n=g.number_of_nodes() 

   infected_nodes=len(infected)

   recovered_nodes=0

   time_stamp=0

  # infected_scale[time_stamp]=infected_scale[time_stamp]+(infected_nodes+recovered_nodes)/n

   infected=spreaders

  # print(infected)

   while(len(infected)>0):

     susceptible_to_infected=[]

     time_stamp=time_stamp+1

     #print("time=",time_stamp)

     #print("infected=",infected)

     for i in infected:

        susceptible=[]

        status[i]=2

        for neighbor in g.neighbors(i):

            if(status[neighbor]==0):

                susceptible.append(neighbor)

        #print("susceptible=",susceptible)        

        total_susceptible=len(susceptible)

        #print("total no of susceptible nodes are=",total_susceptible)

        no_of_susceptible_to_infected=round(beta*total_susceptible)

        #print('after calculating probability=', no_of_susceptible_to_infected)

        while no_of_susceptible_to_infected>0:

            random_index=random.randint(0,total_susceptible-1)

            if susceptible[random_index] not in susceptible_to_infected:

             susceptible_to_infected.append(susceptible[random_index])

             status[susceptible[random_index]]=1

             no_of_susceptible_to_infected=no_of_susceptible_to_infected-1

             #print("infected to be =",susceptible[random_index])

     infected_nodes=len(susceptible_to_infected)

     recovered_nodes=len(infected)

    # print("infected :",infected_nodes)

    # print("recovered:",recovered_nodes)

     ftc=ftc+(recovered_nodes)/n 

    # infected_scale[time_stamp]=infected_scale[time_stamp]+(infected_nodes+recovered_nodes)/n

     infected=susceptible_to_infected  

  

 return  ftc 



def Extended_H_Index(g,h_index):

    ext_h_index={}

    for node in g.nodes():

        sum=h_index[node] 

        for neighbor in g.neighbors(node):

           sum+=h_index[neighbor] 

        ext_h_index[node]=sum

    return ext_h_index
g=CreateGraph()
degree=nx.degree(g)

eigen=Eigen_Vector_Centrality(g)

k_shell=nx.core_number(g)

coreness=Coreness(g,k_shell)

extended_coreness=Extended_Coreness(g,coreness)

h_index=H_Index(g)

extended_h=Extended_H_Index(g,h_index)

cluster=nx.clustering(g)



print('done')
beta=0.045

simulations=100

import os

count=0;

for node in g.nodes():

    spreaders=list()

    spreaders.append(node)

    ftc=SIRModel(g,spreaders,beta,simulations)

   #print(ftc)

    count = count + 1

    if count%1000 == 0:

        os.system('echo '+str(count)+' Files Done')

    with open('dataset.csv','a',newline='') as f:

        row=[]

        row.append(node)

        row.append(degree[node])

        row.append(eigen[node])

        row.append(extended_coreness[node])

        row.append(extended_h[node])

        row.append(cluster[node])

        row.append(ftc)

        w=csv.writer(f)

        w.writerow(row)

        