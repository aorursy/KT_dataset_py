#import libraries

import numpy as np

import matplotlib.pyplot as plt

import random
##Step1: generate random movements for n number of steps



#define number of steps

number_steps = 1000



x,y = 0,0



x_pos = [x]

y_pos = [y]



#filling the coordinates with random movements

for i in range(number_steps):

    direction = random.choice(["N", "S", "W", "E"])

    

    if direction == "N":

        y = y + 1

    elif direction == "S":

        y = y - 1

    elif direction == "W":

        x = x - 1

    else:

        x = x + 1

        

    x_pos.append(x)

    y_pos.append(y)



#Step2: create a plot that summarises all the random movements the agent took

plt.title("Random walk (n = {} steps)".format(number_steps))

plt.plot(x_pos, y_pos)

plt.scatter(x_pos[0], y_pos[0], c="black", label="Start")

plt.scatter(x_pos[-1], y_pos[-1], c="red", label="End")

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()