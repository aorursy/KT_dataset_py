# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# We use the distribution code borrowed from the Kernal - "Plotting Example Gift Weights"

import matplotlib.pyplot as plt

np.random.seed(1234)



class Horse:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(5,2,1)[0])

        self.name = 'horse_' + str(id)



class Ball:

    def __init__(self,id):

        self.weight = max(0, 1 + np.random.normal(1,0.3,1)[0])

        self.name = 'ball_' + str(id)



class Bike:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(20,10,1)[0])

        self.name = 'bike_' + str(id)



class Train:

    def __init__(self,id):

        self.weight = max(0, np.random.normal(10,5,1)[0])

        self.name = 'train_' + str(id)

        

class Coal:

    def __init__(self,id):

        self.weight = 47 * np.random.beta(0.5,0.5,1)[0]

        self.name = 'coal_' + str(id)

        

class Book:

    def __init__(self,id):

        self.weight = np.random.chisquare(2,1)[0]

        self.name = "book_" + str(id)

        

class Doll:

    def __init__(self,id):

        self.weight = np.random.gamma(5,1,1)[0]

        self.name = "doll_" + str(id)



class Block:

    def __init__(self,id):

        self.weight = np.random.triangular(5,10,20,1)[0]

        self.name = "blocks_" + str(id)

        

class Gloves:

    def __init__(self,id):

        self.weight = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

        self.name = "gloves_" + str(id)



def plot_gift(g,i):

    wvec = [x.weight for x in eval(g)]

    plt.figure(i)

    plt.suptitle(g + " = " + str(sum(wvec)))

    plt.hist(wvec)

    

print('finished ')  
# We read in the sample submission file using code borrowed from the kernal - "Shuffle submission.csv."

# The leaderboard score for "sample_submission.csv" is public (9451.44559). 

# As a nice-to-have, we will also use a "light bag" solution formed by removing the last gift from 

# each bag. The leaderboard score for the "light bag" solution is (11238.01386). It's public now.

filename = '../input/sample_submission.csv'

f = open(filename, 'r')

lines = f.read().split("\n")[1:]

if lines[-1] == "":

    lines = lines[:-1]

f.close();

print('finished ')  
import random

random.seed()    # uses system time

prm = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Nine kinds of toy and we don't know the order in which their weights were calculated.

# There are 362880 possible permutations of prm[], so we'll have to be very lucky,

# or very, very patient.

print('finished ')  


random.seed()    # uses system time

for K in range(0, 10000):

    # shuffle the order in which we will calculate the weights

    random.shuffle(prm)

    # Make a wild guess as to which seed was used.

    # Seed must be between 0 and 429496729

    theSeed = random.randrange(10000) # Maybe they picked a low value...

    np.random.seed(theSeed)

    # Assign weights according to the order in prm[]

    for p in range(0,8):

        if prm[p] == 0:

            books = [Book(x) for x in range(1200)]

        if prm[p] == 1:

            horses = [Horse(x) for x in range(1000)]

        if prm[p] == 2:

            bikes = [Bike(x) for x in range(500)]

        if prm[p] == 3:

            trains = [Train(x) for x in range(1000)]

        if prm[p] == 4:

            coals = [Coal(x) for x in range(166)]

        if prm[p] == 5:

            dolls = [Doll(x) for x in range(1000)]

        if prm[p] == 6:

            balls = [Ball(x) for x in range(1100)]

        if prm[p] == 7:

            blocks = [Block(x) for x in range(1000)]

        if prm[p] == 8:

            gloves = [Gloves(x) for x in range(200)]

    # score both sample_submission and "light bag" against these weights

    score = lightScore = 0

    for line in lines:

        bagScore = 0

        toyId = -1

        gifts = line.split(" ")

        if (len(gifts) < 3):

            break

        for gift in gifts:

            lastWeight = bagScore;

            name_id = gift.split("_")

            toyKind = name_id[0]

            toyId = int(name_id[1])

            if toyKind == 'horse':

                bagScore += horses[toyId].weight

            if toyKind == 'ball':

                bagScore += balls [toyId].weight

            if toyKind == 'bike':

                bagScore += bikes [toyId].weight

            if toyKind == 'train':

                bagScore += trains [toyId].weight

            if toyKind == 'coal':

                bagScore += coals [toyId].weight

            if toyKind == 'book':

                bagScore += books [toyId].weight

            if toyKind == 'doll':

                bagScore += dolls [toyId].weight

            if toyKind == 'blocks':

                bagScore += blocks  [toyId].weight

            if toyKind == 'gloves':

                bagScore += gloves[toyId].weight

            lastWeight = bagScore - lastWeight;

        #print(str(bagScore))

        lightBagScore = bagScore - lastWeight;

        if bagScore <= 50.0:

            score += bagScore

        if lightBagScore <= 50.0:

            lightScore += lightBagScore

    # We now have scores for both bags to compare with their public leaderboard scores

    d1 = int(score - 9451.44559)   # Rounded difference between measured and expected

    d2 = int(lightScore - 11238.01386)

    if (abs (score - 9451.44559) < 5.5) and (abs (lightScore - 11238.01386) < 5.5): 

        # Close but not actually useful.

        print(str(theSeed) + " ~~~~~~~~~ errors for sample and light bags " + str(d1) + ",  " + str(d2))

        print(prm)

    if (abs (score - 9451.44559) < 0.005) and (abs (lightScore - 11238.01386) < 0.005): 

        # This would be worth checking out.

        print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print(str(theSeed) + " !!!!!!!!!!!! score = " + str(score))

        print("light bags" + " ~~~~~~~~~~~~~~~~~ score = " + str(lightScore))

        print(prm)

        print("writing file w.txt ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        x = horses + balls + bikes + trains + coals + books + dolls + blocks + gloves

        f = open('w.txt', 'w')

        for i in range(0,len(x)):

            f.write(str(x[i].weight))

            f.write('\n')

        f.close()     

    if (K % 1000) == 0:

        # print something periodically, so we know it's running

        print(str(theSeed) + " errors for sample and light bags " + str(d1) + ",  " + str(d2))

print('              finished ')  