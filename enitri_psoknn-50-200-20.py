import random

from random import shuffle

import math

import time

import numpy as np

import os

from datetime import datetime, timedelta

from pathlib import Path

import pickle

import multiprocessing

from functools import partial



# --- START DEF ALL FUNCTION ----------------------------------------------------------------------+

path_pickle = "../input/"

name_pickle = "current_stage_50-20.pickle"

path_data_train = "../input/data_train_new_stoplist_regex.csv"

path_result_model = "model_result_kaggle_50-20.csv"

fitur_from_csv = []

iterable = []

number_of_process = 600

'''

Fungsi Load Train

'''





def getfile(path):

    with open(path, encoding="utf8", errors='ignore') as f:

        contents = f.read()

    f.close()

    return contents





'''

PSO

'''





# implementasi persamaan 8

def sigmoid(v):

    return 1 / (1 + math.exp(-v))





# implementasi fungsi knn loocv

def knn_loocv(x):

    k = 5

    return LOOCV(items, k, x)





# generate random binary sepanjang D dengan batas 1 sebanyak d

def genBinary(D, d):

    result = []

    for i in range(0, D):

        val = np.random.randint(2, size=1)[0]

        if val == 1 and d > 0:

            result.append(1)

            d -= 1

        else:

            result.append(0)

    return result





# sum matrix

def sumMatrix(X):

    result = 0

    for x in X:

        result += x

    return result





# Objek Partikel

class Particle:

    def __init__(self, x0, y0):

        self.position_i = []  # particle position

        self.velocity_i = []  # particle velocity

        self.pos_best_i = []  # best position individual

        self.fitness_best_i = -1  # best fitness individual

        self.fitness_i = -1  # fitness individual



        for i in range(0, num_dimensions_D):

            self.velocity_i.append(y0[i])

            self.position_i.append(x0[i])



    # evaluate current fitness

    def evaluate(self, costFunc):

        self.fitness_i = costFunc(self.position_i)

        # check to see if the current position is an individual best

        if self.fitness_i > self.fitness_best_i or (

                self.fitness_i == self.fitness_best_i and sumMatrix(self.position_i) < sumMatrix(self.pos_best_i)):

            self.pos_best_i = self.position_i

            self.fitness_best_i = self.fitness_i



    # update new particle velocity and position

    def update_vector(self, pos_best_g, pos_best_it, w, bounds):

        # c1,c2,c3 didapat dari jurnal halaman 44

        c1 = 1.49618

        c2 = 1.49618

        c3 = 0.5

        v_max = 6.0

        v_min = -6.0

        for i in range(0, num_dimensions_D):

            r1 = random.uniform(0, 1)

            r2 = random.uniform(0, 1)

            r3 = random.uniform(0, 1)

            # implementasi persamaan 10

            vel_p = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])

            vel_it = c2 * r2 * (pos_best_it[i] - self.position_i[i])

            vel_g = c3 * r3 * (pos_best_g[i] - self.position_i[i])

            self.velocity_i[i] = w * self.velocity_i[i] + vel_p + vel_it + vel_g

            # implementasi persamaan setelah persamaan 10

            if self.velocity_i[i] > v_max:

                self.velocity_i[i] = v_max

            if self.velocity_i[i] < v_min:

                self.velocity_i[i] = v_min



            S_ij = sigmoid(self.velocity_i[i])

            # Implementasi persamaan 9

            if S_ij > bounds:

                self.position_i[i] = 1

            else:

                self.position_i[i] = 0





class PSO_Continue():

    def __init__(self, fungsi_knnloocv, w_max, w_min, num_particles, maxiter):

        global num_dimensions_D

        # Load Current Stage

        with open(path_pickle + name_pickle, 'rb') as f:

            c_iterasi, c_pos_best_it, c_fitness_best_it, c_pos_best_g, c_fitness_best_g, c_swarm = pickle.load(f)

        # Dimensi D merepresentasikan banyaknya fitur, sesuai jurnal halaman 44

        num_dimensions_D = panjang_fitur

        fitness_best_g = c_fitness_best_g  # best fitness for group

        pos_best_g = c_pos_best_g.copy()  # best position for group

        pos_best_it = c_pos_best_it.copy()  # posisi terbaik dari partikel pada iterasi saat ini

        fitness_best_it = c_fitness_best_it  # Fitness terbaik dari partikel pada iterasi saat ini

        # establish the swarm

        swarm = c_swarm.copy()

        # begin optimization loop

        i = c_iterasi + 1

        panjang_fitur_optimal = 0

        fitur_result_in_key = []

        while i <= maxiter:

            w = w_max - ((w_max - w_min) / maxiter * i)  # implementasi persamaan 5

            # inisial bounds random antara 0 dan 1 contoh 0.5432

            bounds = random.uniform(0, 1)

            # cycle through swarm and update velocities and position

            for j in range(0, num_particles):

                swarm[j].update_vector(pos_best_g, pos_best_it, w, bounds)

            # cycle through particles in swarm and evaluate fitness

            for j in range(0, num_particles):

                swarm[j].evaluate(fungsi_knnloocv)

                # memilih posisi terbaik iterasi

                if swarm[j].fitness_i > fitness_best_it or (swarm[j].fitness_i == fitness_best_it

                                                            and sumMatrix(swarm[j].position_i) < sumMatrix(

                            pos_best_it)):

                    pos_best_it = list(swarm[j].position_i)

                    fitness_best_it = float(swarm[j].fitness_i)

                # determine if current particle is the best (globally)

                if swarm[j].fitness_i > fitness_best_g or (swarm[j].fitness_i == fitness_best_g

                                                           and sumMatrix(swarm[j].position_i) < sumMatrix(pos_best_g)):

                    pos_best_g = list(swarm[j].position_i)

                    fitness_best_g = float(swarm[j].fitness_i)

            os.system(

                "echo " + "Iterasi ke-" + str(i) + " Fitness : " + str(fitness_best_g) + " Posisi : " + str(pos_best_g))

            panjang_fitur_optimal = sumMatrix(pos_best_g)

            iterasi = 0

            for j in pos_best_g:

                if j == 1:

                    fitur_result_in_key.append(fitur_from_csv[iterasi])

                iterasi += 1

            simpan = np.asarray(fitur_result_in_key)

            np.savetxt(path_result_model, simpan, fmt='%s', delimiter=",")

            with open(name_pickle, 'wb') as f:

                pickle.dump([i, pos_best_it, fitness_best_it, pos_best_g, fitness_best_g, swarm], f)

            i += 1

        # print final results

        os.system("echo " + 'Fitur Result in [0,1] :')

        os.system("echo " + str(pos_best_g))

        os.system("echo " + 'Fitur Result in key :')

        os.system("echo " + str(fitur_result_in_key))

        os.system("echo " + 'Fitness Result:')

        os.system("echo " + str(fitness_best_g))

        os.system("echo " + 'Panjang Fitur Terpilih :')

        os.system("echo " + str(panjang_fitur_optimal))





class PSO():

    def __init__(self, fungsi_knnloocv, w_max, w_min, num_particles, maxiter):

        global num_dimensions_D

        # Dimensi D merepresentasikan banyaknya fitur, sesuai jurnal halaman 44

        num_dimensions_D = panjang_fitur

        num_dimension_d = 100

        fitness_best_g = -1  # best fitness for group

        pos_best_g = []  # best position for group

        pos_best_it = []  # posisi terbaik dari partikel pada iterasi saat ini

        fitness_best_it = -1  # Fitness terbaik dari partikel pada iterasi saat ini

        # establish the swarm

        swarm = []

        for i in range(0, num_particles):

            x0 = genBinary(num_dimensions_D, num_dimension_d)

            v0 = [0 for i in range(0, panjang_fitur)]

            swarm.append(Particle(x0, v0))

        for i in range(0, num_particles):

            swarm[i].evaluate(fungsi_knnloocv)

            # memilih posisi terbaik iterasi

            if swarm[i].fitness_i > fitness_best_it or (swarm[i].fitness_i == fitness_best_it

                                                        and sumMatrix(swarm[i].position_i) < sumMatrix(pos_best_it)):

                pos_best_it = list(swarm[i].position_i)

                fitness_best_it = float(swarm[i].fitness_i)

            # determine if current particle is the best (globally)

            if swarm[i].fitness_i > fitness_best_g or (swarm[i].fitness_i == fitness_best_g

                                                       and sumMatrix(swarm[i].position_i) < sumMatrix(pos_best_g)):

                pos_best_g = list(swarm[i].position_i)

                fitness_best_g = float(swarm[i].fitness_i)

        # begin optimization loop

        i = 1

        fitur_result_in_key = []

        panjang_fitur_optimal = 0

        while i <= maxiter:

            w = w_max - ((w_max - w_min) / maxiter * i)  # implementasi persamaan 5

            # inisial bounds random antara 0 dan 1 contoh 0.5432

            bounds = random.uniform(0, 1)

            # cycle through swarm and update velocities and position

            for j in range(0, num_particles):

                swarm[j].update_vector(pos_best_g, pos_best_it, w, bounds)

            # cycle through particles in swarm and evaluate fitness

            for j in range(0, num_particles):

                swarm[j].evaluate(fungsi_knnloocv)

                # memilih posisi terbaik iterasi

                if swarm[j].fitness_i > fitness_best_it or (swarm[j].fitness_i == fitness_best_it

                                                            and sumMatrix(swarm[j].position_i) < sumMatrix(

                            pos_best_it)):

                    pos_best_it = list(swarm[j].position_i)

                    fitness_best_it = float(swarm[j].fitness_i)

                # determine if current particle is the best (globally)

                if swarm[j].fitness_i > fitness_best_g or (swarm[j].fitness_i == fitness_best_g

                                                           and sumMatrix(swarm[j].position_i) < sumMatrix(pos_best_g)):

                    pos_best_g = list(swarm[j].position_i)

                    fitness_best_g = float(swarm[j].fitness_i)

            os.system(

                "echo " + "Iterasi ke-" + str(i) + " Fitness : " + str(fitness_best_g) + " Posisi : " + str(pos_best_g))

            panjang_fitur_optimal = sumMatrix(pos_best_g)

            iterasi = 0

            for j in pos_best_g:

                if j == 1:

                    fitur_result_in_key.append(fitur_from_csv[iterasi])

                iterasi += 1

            simpan = np.asarray(fitur_result_in_key)

            np.savetxt(path_result_model, simpan, fmt='%s', delimiter=",")

            with open(name_pickle, 'wb') as f:

                pickle.dump([i, pos_best_it, fitness_best_it, pos_best_g, fitness_best_g, swarm], f)

            i += 1

        # print final results

        os.system("echo " + 'Fitur Result in [0,1] :')

        os.system("echo " + str(pos_best_g))

        os.system("echo " + 'Fitur Result in key :')

        os.system("echo " + str(fitur_result_in_key))

        os.system("echo " + 'Fitness Result:')

        os.system("echo " + str(fitness_best_g))

        os.system("echo " + 'Panjang Fitur Terpilih :')

        os.system("echo " + str(panjang_fitur_optimal))





'''

KNN LOOCV

'''





# Read data training

def ReadData(fileName):

    # Read the file, splitting by lines

    f = open(fileName, 'r')

    lines = f.read().splitlines()

    f.close()

    # Split the first line by commas,

    # remove the first element and save

    # the rest into a list. The list

    # holds the feature names of the

    # data set.

    features = lines[0].split(',')[:-1]

    # ekstrak fitur

    global fitur_from_csv

    fitur_from_csv = features.copy()

    items = []



    for i in range(1, len(lines)):



        line = lines[i].split(',')



        itemFeatures = {'Class': line[-1]}



        for j in range(len(features)):

            # Get the feature at index j

            f = features[j]



            # Convert feature value to float

            v = float(line[j])



            # Add feature value to dict

            itemFeatures[f] = v



        items.append(itemFeatures)



    shuffle(items)



    return items





def EuclideanDistance(x, y, posisi_fitur):

    # The sum of the squared differences

    # of the elements

    S = 0

    iterasi = 0

    for key in x.keys():

        if posisi_fitur[iterasi] == 1:

            S += math.pow(x[key] - y[key], 2)

        iterasi += 1



        # The square root of the sum

    return math.sqrt(S)





def UpdateNeighbors(neighbors, item, distance, k):

    if len(neighbors) < k:



        # List is not full, add

        # new item and sort

        neighbors.append([distance, item['Class']])

        neighbors = sorted(neighbors)

    else:



        # List is full Check if new

        # item should be entered

        if neighbors[-1][0] > distance:

            # If yes, replace the

            # last element with new item

            neighbors[-1] = [distance, item['Class']]

            neighbors = sorted(neighbors)



    return neighbors





def FindMax(Dict):

    # Find max in dictionary, return

    # max value and max index

    maximum = -1

    classification = ''



    for key in Dict.keys():



        if Dict[key] > maximum:

            maximum = Dict[key]

            classification = key



    return (classification, maximum)





def CalculateNeighborsClass(neighbors, k):

    count = {}



    for i in range(k):

        if neighbors[i][1] not in count:



            # The class at the ith index is

            # not in the count dict.

            # Initialize it to 1.

            count[neighbors[i][1]] = 1

        else:



            # Found another item of class

            # c[i]. Increment its counter.

            count[neighbors[i][1]] += 1



    return count





def Classify(items, n_k, n_x):

    # Hold nearest neighbours. First item

    # is distance, second class

    nItem = items[0]

    Items = items[1]

    # filter 0,1

    itemClass = nItem['Class']

    itemFeatures = {}



    # Get feature values

    for key in nItem:

        if key != 'Class':

            # If key isn't "Class", add

            # it to itemFeatures

            itemFeatures[key] = nItem[key]

    neighbors = []

    for item in Items:

        # Find Euclidean Distance

        distance = EuclideanDistance(itemFeatures, item, n_x)



        # Update neighbors, either adding the

        # current item in neighbors or not.

        neighbors = UpdateNeighbors(neighbors, item, distance, n_k)



        # Count the number of each class

    # in neighbors

    count = CalculateNeighborsClass(neighbors, n_k)



    # Find the max in count, aka the

    # class with the most appearances

    guess = FindMax(count)[0]

    if guess==itemClass:

        return 1

    else:

        return 0



def LOOCV(Item, k, x):

    correct = 0

    arr = []

    res = []

    for j in range(1, len(Item) + 1):

        tempItem = Item.copy()

        nItem = tempItem[j - 1]

        tempItem.pop(j - 1)



        arr.append([nItem,tempItem])

    pool = multiprocessing.Pool(number_of_process)

    func_k = partial(Classify,n_k=k,n_x=x)

    result_list = pool.map(func_k, arr)

    pool.close()

    pool.join()

    correct = sumMatrix(result_list)

    return (correct / len(Item))





# --- END DEF ALL FUNCTION ----------------------------------------------------------------------+

if __name__ == '__main__':

    # --- Start PSO-KNN-LOOCV --- #

    os.system("echo " + "... START FEATURE SELECTION ...")

    os.system("echo " + "... PSO-KNN-LOOCV ...")

    start_time = time.time()

    items = ReadData(path_data_train)

    for j in range(1, len(items) + 1):

        tempItem = items.copy()

        nItem = tempItem[j - 1]

        tempItem.pop(j - 1)

        iterable.append([nItem, tempItem])

    panjang_fitur = len(fitur_from_csv)

    os.system("echo " + "Panjang Fitur " + str(panjang_fitur))

    Wmax = 0.955

    Wmin = 0.5

    current_stage = Path(path_pickle + name_pickle)

    if current_stage.is_file():

        PSO_Continue(knn_loocv, Wmax, Wmin, num_particles=20, maxiter=100)

    else:

        PSO(knn_loocv, Wmax, Wmin, num_particles=20, maxiter=100)

    os.system("echo " + "... FEATURE SELECTION FINISH ...")

    sec = timedelta(seconds=int(time.time() - start_time))

    d = datetime(1, 1, 1) + sec

    os.system("echo " + "Waktu eksekusi : " + str(d.day - 1) + " hari " + str(d.hour) + " jam " + str(

        d.minute) + " menit " + str(d.second) + " detik")
