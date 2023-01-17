# imports

!pip install simanneal

%matplotlib notebook

%matplotlib inline

from scipy.optimize import dual_annealing#, basinhopping, minimize

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from simanneal import Annealer

import random

import json
# load data 

filename = "../input/mlb-stadiums/StadiumsFull.json"

with open(filename, "r") as tsp_data:

    tsp = json.load(tsp_data)

distance_matrix = tsp["DistanceMatrix"]

individual_size = tsp["TourSize"]

teams = tsp["Teams"]

stadium = tsp["Stadiums"]

coords = tsp["Coordinates"]
xy = np.array(coords)

# Plots the tour on a map of the US

# Takes in a tour, xy coordinates, and total distance

def plot_tour(best_tour, xy, best_dist):

    fig = plt.figure()

    fig.set_size_inches(6, 4)

    

    mapUS = Basemap(llcrnrlon=-119,

              llcrnrlat=22,

              urcrnrlon=-64,

              urcrnrlat=49,

              projection='lcc',

              lat_1=32,

              lat_2=45,

              lon_0=-95)



    # load the shape file with "states"

    mapUS.readshapefile('../input/basemap/st99_d00', name='states', drawbounds=True)



    loop_tour = np.append(best_tour, best_tour[0])

    mapUS.plot(xy[:, 0], xy[:, 1], c='r', marker='o', markersize=4, linestyle='')

    lines, = mapUS.plot(xy[loop_tour, 0],

                      xy[loop_tour, 1],

                      c='b',

                      linewidth=1,

                      linestyle='-')

    plt.title('Best Distance {:,d} miles'.format(int(best_dist)))

    plt.show()

    

    

# Prints the information for each step of the tour   

def print_tour(best_tour):

    for s in best_tour:

        print(teams[s], " at ", stadium[s])

        

        

# Prints the convergence plot

def convergence_plot(trajectory, trajectory_best):

    # plot search convergence

    curr = np.array(trajectory)

    best = np.array(trajectory_best)

    fig = plt.figure(figsize=(5, 3.5))

    line_min, = plt.plot(curr[:,0], curr[:,1], label='Curr. Dist.',color='red')

    line_curr, = plt.plot(best[:,0],best[:,1], label='Best. Dist.')

    plt.xlabel('Iteration')

    plt.ylabel('Distance')

    plt.legend(handles=[line_curr, line_min])

    plt.title('Smallest Dist. Found: {:,d} miles'.format(int(best_distSA)));

    plt.show()



# define objective function

def tour_distance(individual, dist_mat):

    distance = dist_mat[individual[-1]][individual[0]]

    for gene1, gene2 in zip(individual[0:-1], individual[1:]):

        distance += dist_mat[gene1][gene2]

    return distance



# reverse a random tour segment

def sub_tour_reversal(tour):

    i, j = np.sort(np.random.choice(individual_size, 2, replace=False))

    swapped = np.concatenate((tour[0:i], tour[j:-individual_size + i - 1:-1],

                              tour[j + 1:individual_size]))

    return [int(swapped[i]) for i in range(individual_size)]



# Random Number Seed

np.random.seed(1)



# initialize with a random tour

current_tour = np.random.permutation(np.arange(individual_size)).tolist()

current_dist = tour_distance(current_tour, distance_matrix)

best_tourSA = current_tour

best_distSA = current_dist





num_moves_no_improve = 0

iteration = 1



####################################################################################

#       These are the variables that can be changed to optimize results

####################################################################################

temp = 2*current_dist  # choose initial temperature around the beginning tour distance

alpha = 0.99 # Used to decay temp

max_moves_no_improve = 5000 # How many moves that can be made without improvement before shutting down





# These help with plotting the search convergence

trajectorySA = [[iteration,current_dist]]

trajectory_bestSA = [[iteration,best_distSA]]







while (num_moves_no_improve < max_moves_no_improve):



    num_moves_no_improve += 1

    new_tour = sub_tour_reversal(current_tour)

    new_dist = tour_distance(new_tour, distance_matrix)

    delta = current_dist - new_dist

    prob = np.exp(min(delta, 0) / temp)

    accept = new_dist < current_dist or np.random.uniform() < prob



    if accept:

        current_tour = new_tour

        current_dist = new_dist

        if current_dist < best_distSA:

            best_tourSA = current_tour

            best_distSA = current_dist

            num_moves_no_improve = 0

    temp *= alpha

    iteration += 1

    trajectorySA.append([iteration,current_dist])

    trajectory_bestSA.append([iteration,best_distSA])

    

convergence_plot(trajectorySA, trajectory_bestSA)
plot_tour(best_tourSA, xy, best_distSA)
print_tour(best_tourSA)