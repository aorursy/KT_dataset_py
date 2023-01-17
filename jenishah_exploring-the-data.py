import os
import numpy as np
import pandas as pd
import glob
from IPython.display import display
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
%matplotlib inline
### Check the number of events ###
ctmp = '../input/train_1/'
print(len(glob.glob(ctmp+'*-hits.csv')))
print(os.listdir('../input/train_1')[:5])
event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))
### number of hits in this event ###
print(len(hits))
display(hits.head(5))
volumes = hits.volume_id.unique()
print((volumes))
g = sns.jointplot(hits.x, hits.y,  s=1, size=12) ## This provides univariate and bivariate plots.
#plt.plot()
g.ax_joint.cla() #clear current axes of sns
plt.sca(g.ax_joint) #set current axes of plt

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.x, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()
particles.head()
# a quick check
pos_hit = sum(particles.nhits.values)
neg_hit = len(particles.nhits.values==0)
print((pos_hit+neg_hit))
print(len(hits)) #the difference may be because of spurious hits
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.distplot(particles.nhits.values, axlabel='Hits/Particle', bins=50)
plt.title('Distribution of number of hits per particle for event 1000.')
plt.subplot(1, 2, 2)
plt.pie(particles.groupby('q')['vx'].count(),
        labels=['negative', 'positive'],
        autopct='%.0f%%',
        shadow=True,
        radius=0.8)
plt.title('Distribution of particle charges.')
plt.show()
truth.head()
# Get particle id with max number of hits in this event
particle = particles.loc[particles.nhits == particles.nhits.max()].iloc[0]
particle2 = particles.loc[particles.nhits == particles.nhits.max()].iloc[1]

# Get points where the same particle intersected subsequent layers of the observation material
p_traj_surface = truth[truth.particle_id == particle.particle_id][['tx', 'ty', 'tz']]
p_traj_surface2 = truth[truth.particle_id == particle2.particle_id][['tx', 'ty', 'tz']]
print(p_traj_surface)
len(particles.loc[particles.nhits == particles.nhits.max()])
# Get particle id with max number of hits in this event
particle = particles.loc[particles.nhits == particles.nhits.max()-5].iloc[19]

# Get points where the same particle intersected subsequent layers of the observation material
p_traj_surface = truth[truth.particle_id == particle.particle_id][['tx', 'ty', 'tz']]

p_traj = (p_traj_surface
          .append({'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}, ignore_index=True)
          .sort_values(by='tz'))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    xs=p_traj.tx,
    ys=p_traj.ty,
    zs=p_traj.tz, marker='o')
ax.plot(
    xs=p_traj_surface.tx,
    ys=p_traj_surface.ty,
    zs=p_traj_surface.tz, marker='o')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z  (mm) -- Detection layers')
plt.title('Trajectories of two particles as they cross the detection surface ($Z$ axis).')
plt.show()
