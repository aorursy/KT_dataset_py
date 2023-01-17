# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import random
%matplotlib notebook
# color values in matplotlib compatible format

yellow_values = [0.788, 0.721, 0.109]

blue_values = [0.113, 0.435, 0.701]

red_values = [1, 0, 0]

black_values = [0, 0, 0]
# define particle class with position velocity and all other

# attributes we want to keep track of when graphing

class ParticleClass(object):



    def __init__(self, position_x, position_y, velocity_x, velocity_y, identity, pid, colors):

        self.position_x = position_x

        self.position_y = position_y

        self.velocity_x = velocity_x

        self.velocity_y = velocity_y

        self.identity = identity

        self.pid = pid

        # this value represents the lifespan of an infection feel free to mess with this

        self.lifespan = 1000 

        self.radius = 0.747

        self.colors = colors # default blue



    def infect(self):

        self.colors = red_values

        self.identity = 'red'



    def heal(self):

        self.colors = yellow_values

        self.identity = 'yellow'

        

    def kill(self):

        self.colors = black_values

        self.velocity_x = 0

        self.velocity_y = 0

        self.identity = 'black'

        

        
# width and height of graph

width, height = 100, 100
# function to create new particles (all with random x, y coordinates and different velocities) 

def make_particle(identity, id, color):

    x_val, y_val = random.randrange(0, width, 1), random.randrange(0, height, 1)

    # here is the velocity for x and y its a random number between -0.3 and 0.3

    # if you increase that range say -1 to 1 the particles will move much faster

    # but the min value does have to be the negative of the max and vice versa

    x_vel, y_vel = random.uniform(-0.35, 0.35), random.uniform(-0.35, 0.35)

    return ParticleClass(x_val, y_val, x_vel, y_vel, identity, id, color)
# function to determine if particles overlap 

def particle_overlap(x1, y1, r1, x2, y2, r2):

    # find distance between particle1 and particle2's x and y coords

    # check if that is less than the sum of their radii 

    a = abs((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    b = (r1 + r2) * (r1 + r2)

    return a < b
my_particles = []

unique_id = 0

# change this for different population size

population_size = 150

# create a ton of particles for graphing

for num in range(population_size + 1):

    # give each particle a unique id so we can check for collisions later

    if num == 5:

        # this will be our patient zero we are giving it red id and

        my_particles.append(make_particle('red', num, red_values))

    else:

        my_particles.append(make_particle('blue', num, blue_values))
# this is where we are going to make all of our 

# changes/update our particles each frame

def update(i1):



    posx = []

    posy = []

    colors = []



    for i in range(0, len(my_particles)):

        # check for intersecting particles by creating nested for loop

        for j in range(0, len(my_particles)):

            # make sure we are not comparing the same particle

            if my_particles[i].pid != my_particles[j].pid:

                # create condition to check if particles overlap

                overlap = particle_overlap(my_particles[i].position_x, my_particles[i].position_y, my_particles[i].radius,

                                                    my_particles[j].position_x, my_particles[j].position_y,

                                                    my_particles[j].radius)



                # only red particles should be able to infect blue ones

                # in this simulation everyone else will be dead or immune

                if overlap and my_particles[i].identity + my_particles[j].identity == 'redblue':

                    # so this is where we determine what percentage of the population is vulnerable to infection

                    # at the momenent it is half but you can change the denominator to whatever you want

                    if my_particles[j].pid < population_size / 2:

                        my_particles[i].infect()

                        my_particles[j].infect()

                        

        # give particles velocity

        my_particles[i].position_x += my_particles[i].velocity_x

        my_particles[i].position_y += my_particles[i].velocity_y



        # change position and velocity when particles hit walls

        # right wall boundaries.

        if my_particles[i].position_x + my_particles[i].radius > width:

            my_particles[i].position_x = width - my_particles[i].radius

            my_particles[i].velocity_x *= -1



        # left wall boundaries.

        if my_particles[i].position_x - my_particles[i].radius < 0:

            my_particles[i].position_x = my_particles[i].radius

            my_particles[i].velocity_x *= -1



        # bottom wall boundaries.

        if my_particles[i].position_y - my_particles[i].radius < 0:

            my_particles[i].position_y = my_particles[i].radius

            my_particles[i].velocity_y *= -1



        # top wall boundaries.

        if my_particles[i].position_y >= height:

            my_particles[i].position_y = height - my_particles[i].radius

            my_particles[i].velocity_y *= -1

        

        # heal infected particles

        # heal or kill infected particles

        if my_particles[i].identity == 'red':

            my_particles[i].lifespan -= 1

            if my_particles[i].lifespan == 0:

                if random.uniform(0, 1) <= 0.04:

                    my_particles[i].kill()

                else:

                    my_particles[i].heal()

        

        

        posx.append(my_particles[i].position_x)

        posy.append(my_particles[i].position_y)

        colors.append(my_particles[i].colors)



    # set x y coordinates and particle colors for our scatter plots

    particle_scat.set_offsets(np.vstack((posx, posy)).T)

    particle_scat.set_color(colors)



    return particle_scat,

# create our particle scatter plot

fig, ax = plt.subplots()

ax.set_xlim(0, width)

ax.set_ylim(0, height)

particle_scat = ax.scatter(0, 0, linewidth=1, marker = 'o', s=40)



# create dummy values for legend

red_dot = ax.scatter(-2, 0, s=20, linewidth=1, c=np.array(red_values).reshape(1,3))

blue_dot = ax.scatter(-2, 0, s=20, linewidth=1, c=np.array(blue_values).reshape(1, 3))

yellow_dot = ax.scatter(-2, 0, s=20, linewidth=1, c=np.array(yellow_values).reshape(1, 3))

black_dot = ax.scatter(-2, 0, s=20, linewidth=1, c='black')



# create legend

plt.legend(handles=[blue_dot, red_dot, yellow_dot, black_dot], 

           labels=['Non-infected', 'Infected', 'Recovered/Immune',

                   'Deceased'], loc="lower right")



plt.title("Pandemic Growth (Assuming 0.15 Infection Rate)")

plt.axis('off')



fig.set_size_inches(9.5, 6.5, forward=True)

# graph 

scat_ani = FuncAnimation(fig, func=update, interval=10, blit=True)

plt.show()