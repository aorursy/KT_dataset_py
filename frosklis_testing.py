%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import sys
import traceback

# Data
datapath = "../input/gifts.csv"

import pandas as pd
from haversine import haversine
import itertools

allgifts = pd.read_csv(datapath, index_col="GiftId")
import math
import random
import sys

class Ant():
    def __init__(self, ID, start_node, colony):
        self.ID = ID
        self.start_node = start_node
        self.colony = colony
        
        self.weight = self.colony.graph.weight

        self.curr_node = self.start_node
        self.graph = self.colony.graph
        self.path_vec = []
        self.path_vec.append(self.start_node)
        self.path_cost = 0

        # same meaning as in standard equations
        self.Beta = 1
        #self.Q0 = 1  # Q0 = 1 works just fine for 10 city case (no explore)
        self.Q0 = 0.5
        self.Rho = 0.99

        # store the nodes remaining to be explored here
        self.nodes_to_visit = {}

        for i in range(0, self.graph.num_nodes):
            if i != self.start_node:
                self.nodes_to_visit[i] = i

        # create n X n matrix 0'd out to start
        self.path_mat = []

        for i in range(0, self.graph.num_nodes):
            self.path_mat.append([0]*self.graph.num_nodes)

    # overide Thread's run()
    def run(self):
        graph = self.colony.graph
        while not self.end():
            # we need exclusive access to the graph
            graph.lock.acquire()
            new_node = self.state_transition_rule(self.curr_node)
            self.path_cost += graph.delta(self.curr_node, new_node) * self.weight
            self.weight -= graph.nodeWeight(new_node)

            self.path_vec.append(new_node)
            self.path_mat[self.curr_node][new_node] = 1  #adjacency matrix representing path
            
            self.local_updating_rule(self.curr_node, new_node)
            graph.lock.release()

            self.curr_node = new_node
            
            # print self.ID, self.path_vec

        # don't forget to close the tour
        self.path_cost += graph.delta(self.path_vec[-1], self.path_vec[0]) * self.weight
        self.weight -= graph.nodeWeight(self.path_vec[0])
        self.path_vec.append(self.path_vec[0])

        # send our results to the colony
        self.colony.update(self)

        # allows thread to be restarted (calls Thread.__init__)
        self.__init__(self.ID, self.start_node, self.colony)

    def end(self):
        return (not self.nodes_to_visit) or self.path_cost >= self.colony.best_path_cost

    # described in report -- determines next node to visit after curr_node
    def state_transition_rule(self, curr_node):
        graph = self.colony.graph
        q = random.random()
        max_node = -1
        if q < self.Q0:
            # print "Exploitation"
            max_val = -1
            val = None

            for node in self.nodes_to_visit.values():
                if graph.tau(curr_node, node) == 0:
                    raise Exception("tau = 0")

                val = graph.tau(curr_node, node) * math.pow(graph.etha(curr_node, node), self.Beta)
                if val > max_val:
                    max_val = val
                    max_node = node
        else:
            # print "Exploration"
            sum = 0
            node = -1
            v = self.nodes_to_visit.values()
            max_node = v[random.randint(0, len(v)-1)]
        
        if max_node < 0:
            raise Exception("max_node < 0")

        del self.nodes_to_visit[max_node]
        
        return max_node

    # phermone update rule for indiv ants
    def local_updating_rule(self, curr_node, next_node):
        graph = self.colony.graph
        val = (1 - self.Rho) * graph.tau(curr_node, next_node) + (self.Rho * graph.tau0)
        graph.update_tau(curr_node, next_node, val)


from threading import Lock

import random
import sys

class AntColony:
    def __init__(self, graph, num_ants, num_iterations):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.Alpha = 0.1

        self.reset()

    def reset(self):
        self.best_path_cost = 1e1000
        self.best_path_vec = None
        self.best_path_mat  = None
        self.last_best_path_iteration = 0

    def start(self):
        self.ants = self.create_ants()
        self.iter_counter = 0

        while self.iter_counter < self.num_iterations:
            self.iteration()

            lock = self.graph.lock
            lock.acquire()
            self.global_updating_rule()
            lock.release()

    # one iteration involves spawning a number of ant threads
    def iteration(self):
        self.avg_path_cost = 0
        self.ant_counter = 0
        self.iter_counter += 1
        # print "iter_counter = %s" % (self.iter_counter,)
        for ant in self.ants:
            # print "starting ant = %s" % (ant.ID)
            ant.run()

    def num_ants(self):
        return len(self.ants)

    def num_iterations(self):
        return self.num_iterations

    def iteration_counter(self):
        return self.iter_counter

    # called by individual ants
    def update(self, ant):
        lock = Lock()
        lock.acquire()

        self.ant_counter += 1

        self.avg_path_cost += ant.path_cost

        # book-keeping
        if ant.path_cost < self.best_path_cost:
            self.best_path_cost = ant.path_cost
            self.best_path_mat = ant.path_mat
            self.best_path_vec = ant.path_vec
            self.last_best_path_iteration = self.iter_counter

        if self.ant_counter == len(self.ants):
            self.avg_path_cost /= len(self.ants)
        lock.release()

    def done(self):
        return self.iter_counter == self.num_iterations

    # assign each ant a random start-node
    def create_ants(self):
        self.reset()
        ants = []
        for i in range(0, self.num_ants):
            # ant = Ant(i, random.randint(0, self.graph.num_nodes - 1), self)
            ant = Ant(i, 0, self)
            ants.append(ant)
        
        return ants

    # changes the tau matrix based on evaporation/deposition 
    def global_updating_rule(self):
        evaporation = 0
        deposition = 0

        for r in range(0, self.graph.num_nodes):
            for s in range(0, self.graph.num_nodes):
                if r != s:
                    delt_tau = self.best_path_mat[r][s] / self.best_path_cost
                    evaporation = (1 - self.Alpha) * self.graph.tau(r, s)
                    deposition = self.Alpha * delt_tau

                    self.graph.update_tau(r, s, evaporation + deposition)

from threading import Lock
from haversine import haversine

class AntGraph:
    def __init__(self, nodes, tau_mat=None):
        self.num_nodes = len(nodes)
        self.nodes = nodes
        
        # build the matrix
        delta_mat = {}
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                delta_mat[(i,j)] = haversine((nodes.iloc[i].Latitude, nodes.iloc[i].Longitude), (nodes.iloc[j].Latitude, nodes.iloc[j].Longitude))
                delta_mat[(j,i)] = delta_mat[(i,j)]
        self.delta_mat = delta_mat # matrix of node distance deltas
        
        self.weight = nodes.Weight.sum()
        self.lock = Lock()
        # tau mat contains the amount of phermone at node x,y
        if tau_mat is None:
            self.tau_mat = []
            for i in range(0, self.num_nodes):
                self.tau_mat.append([0]*self.num_nodes)

    def nodeWeight(self, node):
        return self.nodes.iloc[node].Weight
    
    def delta(self, r, s):
        if r == s:
            return 0
        return self.delta_mat[(r,s)]

    def tau(self, r, s):
        return self.tau_mat[r][s]

    # 1 / delta = eta or etha 
    def etha(self, r, s):
        return 1.0 / self.delta(r, s)

    # inner locks most likely not necessary
    def update_tau(self, r, s, val):
        lock = Lock()
        lock.acquire()
        self.tau_mat[r][s] = val
        lock.release()

    def reset_tau(self):
        lock = Lock()
        lock.acquire()
        avg = self.average_delta() * self.weight / self.num_nodes

        # initial tau 
        self.tau0 = 1.0 / (self.num_nodes * 0.5 * avg)
        
        for r in range(0, self.num_nodes):
            for s in range(0, self.num_nodes):
                self.tau_mat[r][s] = self.tau0
        lock.release()

    # average delta in delta matrix
    def average_delta(self):
        return self.average(self.delta)

    # average tau in tau matrix
    def average_tau(self):
        return self.average(self.tau)

    # average val of a matrix
    def average(self, matrix):
        sum = 0
        for r in range(0, self.num_nodes):
            for s in range(0, self.num_nodes):
                sum += matrix(r,s)

        avg = sum / (self.num_nodes * self.num_nodes)
        return avg


def optimize(nodes ):
    num_ants = 10
    num_iterations = 10
    num_repetitions = 2

    best_path_vec = None
    best_path_cost = 1e1000
    for i in range(0, num_repetitions):
        graph = AntGraph(nodes)
        graph.reset_tau()
        ant_colony = AntColony(graph, num_ants, num_iterations)
        ant_colony.start()
        if ant_colony.best_path_cost < best_path_cost:
            best_path_vec = ant_colony.best_path_vec
            best_path_cost = ant_colony.best_path_cost
        
    return (best_path_vec, best_path_cost)
%%time
from sklearn.cluster import KMeans
data = allgifts
data.loc[:,"cluster"] = 0
train = data.loc[:,["Latitude","Longitude"]]

flag = True
i = 0
while flag:
    i += 1
    flag = False
    clusters = data["cluster"].unique()
    if i % 10 <= 1:
        print("Iter: %3d, Clusters: %d" % (i, len(clusters)))
    for clus in clusters:
        indices = data["cluster"] == clus
        weight = data.loc[indices, :]["Weight"].sum()
        if weight < 1000:
            continue
        flag = True
        # print "Cluster: %8d Weight: %f" % (int(clus), weight)
        t = train.loc[indices,:]
        c = KMeans(init='k-means++', n_clusters=2, n_init=10).fit_predict(t)
        data.loc[indices, "cluster"] = c + 2*data["cluster"].loc[indices]
    

# Build the nodes vector
nodes = []
north_pole = pd.DataFrame({'Latitude': [0],
                           'Longitude': [0],
                           'Weight': [10]},
                          index=[0])
for c in data["cluster"].unique():
    d = data.loc[data["cluster"] == c,]
    nodes.append(pd.concat([north_pole, d]))
    
print( len(nodes))
def writeSolution(sol, path="data/solution_1.csv"):
    with open(path, "w") as f:
        f.write("GiftId,TripId\n")
        for i, t in enumerate(sol):
            for g in list(t[0]):
                if g == 0:
                    continue
                f.write("%d,%d\n" % (g,i+1))

from multiprocessing import Pool
p = Pool(50)
results = p.map(optimize, nodes)

writeSolution(results, path="data/sol_002.csv")