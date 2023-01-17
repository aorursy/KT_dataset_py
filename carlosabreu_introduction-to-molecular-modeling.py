#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Created by Carlos Abreu - April 2020'''
import random
import numpy as np
import matplotlib.pyplot as plt


# In[81]:



n_particles = 100
n_steps = 500
dt = 0.001
width = 10.0

#Define Coordinates

def initial_position(x_coord, y_coord):
    x_coord = [np.random.random()*width for i in range(n_particles)]
    y_coord = [np.random.random()*width for i in range(n_particles)]
    return x_coord, y_coord
x_coord, y_coord = initial_position(x_coord, y_coord)


# In[104]:


plt.plot(x_coord, y_coord, 'bs')
plt.axis((-width, width, -width, width))
plt.quiver(x_coord, y_coord)
plt.show()


# In[105]:


#Define Velocities

def initial_velocities(x_vel, y_vel):
    x_vel = [2*(np.random.random()-0.5)*width for i in range(n_particles)]
    y_vel = [2*(np.random.random()-0.5)*width for i in range(n_particles)]
    return x_vel, y_vel
x_vel, y_vel = initial_velocities(x_vel, y_vel)


# In[106]:


plt.plot(x_vel, y_vel, 'ro')
plt.axis((-width, width, -width, width))
plt.quiver(x_coord, y_coord, x_vel, y_vel)
plt.show()


# In[107]:


#Iterate over random steps taken per molecule

def random_walk(x_coord, y_coord, x_vel, y_vel):
    for i in range(n_particles):
                x_coord[i] += x_vel[i]*dt
                y_coord[i] += y_vel[i]*dt
        
    if abs(x_coord[i]) > width:
            x_vel[i] = -x_vel[i]
            x_coord[i] += x_vel[i]*dt
    
    if abs(y_coord[i]) > width:
            y_vel[i] = -y_vel[i]
            y_coord[i] += y_vel[i]*dt
    return x_coord, y_coord, x_vel, y_vel

for i in range(n_steps):
    x_coord, y_coord, x_vel, y_vel = random_walk(x_coord, y_coord, x_vel, y_vel)
    
#Save a frame every 10 steps
    if i%10 ==0:
        add_frame(x_coord, y_coord, i)


# In[108]:


plt.plot(x_coord, y_coord, x_vel, y_vel, 'ro')
plt.axis((-width, width, -width, width))
plt.show()


# In[109]:


def add_frame(xs,ys,i):
  global trajectory
  if i == 0:
    trajectory = ''
  trajectory += str(n_particles) + '\ntitle\n'
  for x, y in zip(xs,ys):
    trajectory += ' '.join(['Ar',str(x),str(y),'0.0\n'])


# In[110]:


plt.plot(x_coord, y_coord, 'bs')
plt.axis((-width, width, -width, width))
plt.show()



# In[117]:


#Model Simulation

import py3Dmol as p3D
view = p3D.view()
view.addModelsAsFrames(trajectory,'xyz')
view.setStyle({'sphere': {'color':'magenta'}})
view.animate({'loop':'forward'},viewer=(0,0))
view.zoomTo()


#