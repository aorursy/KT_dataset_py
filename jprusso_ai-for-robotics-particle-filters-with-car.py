from math import *

import random

# --------

# 

# the "world" has 4 landmarks.

# the robot's initial coordinates are somewhere in the square

# represented by the landmarks.

#

# NOTE: Landmark coordinates are given in (y, x) form and NOT

# in the traditional (x, y) format!



landmarks  = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] # position of 4 landmarks in (y, x) form.

world_size = 100.0 # world is NOT cyclic. Robot is allowed to travel "out of bounds"

max_steering_angle = pi/4 # You don't need to use this value, but it is good to keep in mind the limitations of a real car.
# ------------------------------------------------

# 

# this is the robot class

#



class robot:



    # --------



    # init: 

    #	creates robot and initializes location/orientation 

    #



    def __init__(self, length = 10.0):

        self.x = random.random() * world_size # initial x position

        self.y = random.random() * world_size # initial y position

        self.orientation = random.random() * 2.0 * pi # initial orientation

        self.length = length # length of robot

        self.bearing_noise  = 0.0 # initialize bearing noise to zero

        self.steering_noise = 0.0 # initialize steering noise to zero

        self.distance_noise = 0.0 # initialize distance noise to zero

    

    def __repr__(self):

        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    # --------

    # set: 

    #	sets a robot coordinate

    #



    def set(self, new_x, new_y, new_orientation):



        if new_orientation < 0 or new_orientation >= 2 * pi:

            raise(ValueError, 'Orientation must be in [0..2pi]')

        self.x = float(new_x)

        self.y = float(new_y)

        self.orientation = float(new_orientation)





    # --------

    # set_noise: 

    #	sets the noise parameters

    #



    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):

        # makes it possible to change the noise parameters

        # this is often useful in particle filters

        self.bearing_noise  = float(new_b_noise)

        self.steering_noise = float(new_s_noise)

        self.distance_noise = float(new_d_noise)

    

    # --------

    # move:

    #   move along a section of a circular path according to motion

    #

    

    def move(self, motion): # Do not change the name of this function

        alfa = motion[0]

        distance = motion[1]

        

        b = (distance / self.length) * tan(alfa)

        

        if b > 0.001:

            r = distance / b

            cx = self.x - sin(self.orientation) * r

            cy = self.y + cos(self.orientation) * r

            x = cx + sin(self.orientation + b) * r

            y = cy - cos(self.orientation + b) * r

        else:

            x = self.x + distance * cos(self.orientation)

            y = self.y + distance * sin(self.orientation)

            

        orientation = (self.orientation + b) % (2 * pi)

        result = robot(self.length)

        result.set(x, y, orientation)

        

        return result

    

    def sense(self, add_noise = 1):

        Z = []

        for i in range(len(landmarks)):

            bearing = atan2(landmarks[i][0] - self.y, landmarks[i][1] - self.x) - self.orientation

            

            if add_noise:

                bearing += random.gauss(0.0, self.bearing_noise)

                

            bearing %= 2.0 * pi

            Z.append(bearing)

        return Z
length = 20.

bearing_noise  = 0.0

steering_noise = 0.0

distance_noise = 0.0



myrobot = robot(length)

myrobot.set(0.0, 0.0, 0.0)

myrobot.set_noise(bearing_noise, steering_noise, distance_noise)



motions = [[0.2, 10.] for row in range(10)]



T = len(motions)



print('Robot:    ', myrobot)

for t in range(T):

    myrobot = myrobot.move(motions[t])

    print('Robot:    ', myrobot)
length = 20.

bearing_noise  = 0.0

steering_noise = 0.0

distance_noise = 0.0



myrobot = robot(length)

myrobot.set(0.0, 0.0, 0.0)

myrobot.set_noise(bearing_noise, steering_noise, distance_noise)



motions = [[0.0, 10.0], [pi / 6.0, 10], [0.0, 20.0]]



T = len(motions)



print('Robot:    ', myrobot)

for t in range(T):

    myrobot = myrobot.move(motions[t])

    print('Robot:    ', myrobot)
##

## 1) The following code should print the list [6.004885648174475, 3.7295952571373605, 1.9295669970654687, 0.8519663271732721]

##

##

length = 20.

bearing_noise  = 0.0

steering_noise = 0.0

distance_noise = 0.0



myrobot = robot(length)

myrobot.set(30.0, 20.0, 0.0)

myrobot.set_noise(bearing_noise, steering_noise, distance_noise)



print('Robot:        ', myrobot)

print('Measurements: ', myrobot.sense())