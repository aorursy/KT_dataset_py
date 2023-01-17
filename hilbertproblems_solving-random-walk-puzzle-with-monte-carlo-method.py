#import libraries

import random

import pandas as pd



def random_walk(n):

    """

    input: n which represents the number of blocks to walk

    output: return coordinates after n blocks random walk

    """

    

    #initialize the initial coordinates that define the 

    #position of the agent prior to taking a random walk

    x, y = 0, 0

    

    for i in range(n):

        step = random.choice(["N", "S", "E", "W"])

        

        if step == "N":

            y = y + 1

        elif step == "S":

            y = y - 1

        elif step == "E":

            x = x + 1

        else:

            x = x - 1

    return (x, y)
for i in range(20):

    walk = random_walk(10)

    print(walk, "Distance from starting location = ", abs(walk[0]) + abs(walk[1])) 
percentages = dict()

total_walks = 10000

for walk_length in range(1, 31):

    

    no_trasport_counter = 0 #Number of walks 4 or fewer from home

    

    for i in range(total_walks):

        (x, y) = random_walk(walk_length)

        #compute diatnce from home (starting location)

        distance = abs(x) + abs(y)

        #check if distance is 4 blocks or fewer

        if distance <= 4:

            no_trasport_counter += 1

            

    #compute percentage of random walks that ended with a walk back home

    no_transport_percentage = (float(no_trasport_counter) / total_walks) * 100

     

    percentages[walk_length] = no_transport_percentage

        

    #print("Random walk size = ", walk_length,

          #" / % of no transport = ", no_transport_percentage)
table = pd.DataFrame(list(percentages.items()), columns=["Walk size (nuber of blocks)", "% no transport"])



#filter out rows where "% no transport" < 50%

table = table[table["% no transport"] >= 50.0]

table