from numpy import *

#import tensorflow as tf

from numpy.random import *

from matplotlib.pyplot import *

from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go



#this is the inverse of the cumulutative function of the Henney-Green phase function

def u(X1,g):

  A = (1-2*g/(1+g**2))**(-1/2)

  B = X1*2*g*(1+g**2)**(3/2)

  C = (1-g**2)*(1+g**2)

  D = A - B/C

  E = 1-D**(-2)

  F = (1+g**2)*E/2/g

  return(arccos(F))



def get_angle(g):

  r = rand()

  return(u(r,g))



def propagate(mu_t):

  r = tf.random.uniform([1])

  s = -tf.math.log(1-r)/mu_t

  return(s[0])
def SIMULATE_LASER(mu_a, mu_s, g, INITIAL):

  mu_t = mu_s+mu_a

  a    = mu_s/mu_t



  POS = [[0,0,0]]*len(INITIAL)



  s0 = array([0,0])



  i=0

  for R in INITIAL:

    s = s0



    while True:

      direction = array([sin(s[0])*cos(s[1]), sin(s[0])*sin(s[1]), cos(s[0])])

      R = R + propagate(mu_t)*direction

  

      #Absorption

      if rand()>a: 

        POS[i] = array([R[0],R[1],R[2]]);

        break



      #Scatter

      dphi = rand()*2*pi

      dtheta = get_angle(g)

      s = s + array([dtheta, dphi])

    i+=1

  return(array(POS))
mu_s = 0.001

mu_a = 0.1



#print(mu_s/(mu_s+mu_a))
#foScatter = SIMULATE_LASER(mu_a, mu_s, 0.5, [[0,0,0]]*500) #very low scattering
#X,Y,Z = foScatter[:,0],foScatter[:,1],foScatter[:,2]

#Data = go.Scatter3d(x=X, y=Y, z=Z, 

#                    marker = dict(size=1,color=Z,       colorscale='Viridis',), 

#                    line   = dict(       color='white', width=0.001))

#fig = go.Figure(data = Data)

#fig.show()
soScatter = SIMULATE_LASER(10, 40, 0.955, [[0,0,0]]*3000)
scatter(soScatter[:,2], soScatter[:,0], s=0.1); 

xlim(0);xlabel('z-axis'); ylabel('y-axis'); title('Projection onto y-z plane');
X,Y,Z = soScatter[:,0],soScatter[:,1],soScatter[:,2]



Data = go.Scatter3d(x=X, y=Y, z=Z, 

                    marker = dict(size=1,color=Z,       colorscale='Viridis',), 

                    line   = dict(       color='white', width=0.001))

fig = go.Figure(data = Data)

fig.show()
soScatter1 = SIMULATE_LASER(1, 100, 0.998, [[0,0,0]]*500)
scatter(soScatter1[:,2], soScatter1[:,0], s=0.1); 

xlim(0);xlabel('z-axis'); ylabel('y-axis'); title('Projection onto y-z plane');
X,Y,Z = soScatter1[:,0],soScatter1[:,1],soScatter1[:,2]



Data = go.Scatter3d(x=X, y=Y, z=Z, 

                    marker = dict(size=1,color=Z,       colorscale='Viridis',), 

                    line   = dict(       color='white', width=0.001))

fig = go.Figure(data = Data)

fig.show()