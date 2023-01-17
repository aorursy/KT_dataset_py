from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 

from IPython.display import HTML
from matplotlib import animation
import io, base64
# plt.style.use('fivethirtyeight')

from plotly.offline import init_notebook_mode, iplot
from matplotlib.pyplot import pie, show
import plotly.graph_objs as go
import numpy as np

# from trackml.dataset import load_event, load_dataset
# from trackml.score import score_event

sns.set_style("whitegrid", {'axes.grid' : False})
init_notebook_mode(connected=True)
train_path = '../input/train_1/'
train_files = os.listdir(train_path)
train_files[:10]
len(train_files)
cells_df, truth_df, particles_df, hits_df = [], [], [], []
for filename in sorted(train_files)[:20]:
    if "cells" in filename:
        cell_df = pd.read_csv(train_path+filename)
        cells_df.append(cell_df)
    elif "hits" in filename:
        hit_df = pd.read_csv(train_path+filename)
        hits_df.append(hit_df)
    elif "particles" in filename:
        particle_df = pd.read_csv(train_path+filename)
        particles_df.append(particle_df)
    elif "truth" in filename:
        trut_df = pd.read_csv(train_path+filename)
        truth_df.append(trut_df)
# cells_df[0].shape
cells_df[0].head(10)
def dist(df, col, bins, color, title, kde=False):
    plt.figure(figsize=(15,3))
    sns.distplot(df[col].values, bins=bins, color=color, kde=kde)
    plt.title(title, fontsize=14);
    plt.show();

def mdist(df, col, bins, color, kde=False):
    f, axes = plt.subplots(1, 2, figsize=(15,3))
    sns.distplot(df[1][col].values, bins=bins, color=color, rug=False, ax=axes[0], kde=kde)
    sns.distplot(df[2][col].values, bins=bins, color=color, rug=False, ax=axes[1], kde=kde)

    f, axes = plt.subplots(1, 2, figsize=(15,3))
    sns.distplot(df[3][col].values, bins=bins, color=color, rug=False, ax=axes[0], kde=kde)
    sns.distplot(df[4][col].values, bins=bins, color=color, rug=False, ax=axes[1], kde=kde)

dist(cells_df[0], 'value', 10, 'red', 'cells.value')
mdist(cells_df, 'value', 10, 'green', kde=True)
dist(cells_df[0], 'ch0', 100, 'red', 'cells.ch0')
# cells_df[0].ch0.describe()
# cells_df[0].ch0.value_counts()
mdist(cells_df, 'ch0', 10, 'green', kde=True)
dist(cells_df[0], 'ch1', 100, 'red', 'cells.ch1')
# cells_df[0].ch1.describe()

mdist(cells_df, 'ch1', 10, 'green', kde=True)
ch0df = cells_df[0].groupby('ch0').agg({'value' : 'mean'}).reset_index()
ch1df = cells_df[0].groupby('ch1').agg({'value' : 'mean'}).reset_index()

f, axes = plt.subplots(2, 1, figsize=(15,10))
sns.regplot(x='ch0', y='value', data=ch0df, fit_reg=False, color='#ff4c64', ax=axes[0])
sns.regplot(x='ch1', y='value', data=ch1df, fit_reg=False, color='#89ea7c', ax=axes[1])
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
hits_df[0].head(10)
# f, axes = plt.subplots(1, 2, figsize=(15,5));
# sns.distplot(hits_df[0].x.values, color='red', rug=False, ax=axes[0])
# sns.distplot(hits_df[0].y.values, color='red', rug=False, ax=axes[1])
# axes[0].set_title("distribution of x coordinate of particles");
# axes[1].set_title("distribution of y coordinate of particles");

# f, axes = plt.subplots(1, 2, figsize=(15,5));
# sns.distplot(hits_df[0].z.values, color='red', rug=False, ax=axes[0])
# sns.regplot(x=hits_df[0][:2000].x.values, y=hits_df[0][:2000].y.values, fit_reg=False, color='#ff4c64', ax=axes[1])
# axes[0].set_title("distribution of z coordinate of particles");
# axes[1].set_title("plotting of x and y of particles");
# axes[1].set(xlabel='x', ylabel='y');

hits_small = hits_df[0][['x','y','z']]
sns.pairplot(hits_small, palette='husl', size=6)
plt.show()
dist(hits_df[0], 'volume_id', 20, 'red', 'hits.volume_id distribution', kde=True)
# hits_df[0].volume_id.value_counts()
mdist(hits_df, 'volume_id', 20, 'green')
dist(hits_df[0], 'layer_id', 20, 'red', 'hits.layer_id distribution', kde=False)
dist(hits_df[0], 'module_id', 100, 'green', 'hits.module_id distribution')
hits_df[0]['origin_dis'] = np.sqrt(np.square(hits_df[0].x) + np.square(hits_df[0].y) + np.square(hits_df[0].z))
dist(hits_df[0], 'origin_dis', 100, 'red', 'hits.origin_dis distribution')
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure(figsize=(10,8))
# plt.figure();

ax = Axes3D(fig)

hits_sample = hits_df[0].sample(3000)
ax.scatter(hits_sample.x, hits_sample.y, hits_sample.z)
pyplot.show()
# truth_df[0][truth_df[0]['particle_id'] == 414345390649769984]
# particle_df.sort_values(['nhits'])[particle_df.nhits == 12]
# 4513357793067008, 112595763120308224

vals = list(truth_df[0][truth_df[0]['particle_id'] == 112595763120308224].hit_id.values)
tempdf = hits_df[0][hits_df[0]['hit_id'].isin(vals)]
tempdf
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
plt.style.use('fivethirtyeight')
def animate(hit_id):
    ax.set_xlim([0,900])
    ax.set_ylim([-600,300])
    ax.set_zlim([-300,-10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
#     x = tempdf.x[tempdf.hit_id == hit_id]
#     y = tempdf.y[tempdf.hit_id == hit_id]
#     z = tempdf.z[tempdf.hit_id == hit_id]
    
    
    x = tempdf.x[tempdf.hit_id == hit_id]
    y = tempdf.y[tempdf.hit_id == hit_id]
    z = tempdf.z[tempdf.hit_id == hit_id]
    s = tempdf.module_id[tempdf.hit_id == hit_id] 
#     c = tempdf.layer_id[tempdf.hit_id == hit_id]
    ax.scatter(x, y, z, s=300)# 'o', color = 'r', markersize = 10, alpha = 0.5) 
    
ani = animation.FuncAnimation(fig, animate, tempdf.hit_id.unique().tolist())
ani.save('animation1.gif', writer='imagemagick', fps=2)


# truth_df[0][truth_df[0]['particle_id'] == 414345390649769984]
# particle_df.sort_values(['nhits'])[particle_df.nhits == 12]
# 4513357793067008, 112595763120308224

vals = list(truth_df[0][truth_df[0]['particle_id'] == 968275088115761152].hit_id.values)
tempdf = hits_df[0][hits_df[0]['hit_id'].isin(vals)]
# tempdf
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
plt.style.use('fivethirtyeight')
def animate(hit_id):
#     ax.set_xlim([-60,20])
#     ax.set_ylim([0,800])
#     ax.set_zlim([-100,-3000])

    ax.set_xlim([20,1100])
    ax.set_ylim([-110,0])
    ax.set_zlim([0,1300])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
#     x = tempdf.x[tempdf.hit_id == hit_id]
#     y = tempdf.y[tempdf.hit_id == hit_id]
#     z = tempdf.z[tempdf.hit_id == hit_id]
    
    
    x = tempdf.x[tempdf.hit_id == hit_id]
    y = tempdf.y[tempdf.hit_id == hit_id]
    z = tempdf.z[tempdf.hit_id == hit_id]
    ax.scatter(x, y, z, s=300)# 'o', color = 'r', markersize = 10, alpha = 0.5)
    
ani = animation.FuncAnimation(fig, animate, tempdf.hit_id.unique().tolist())
ani.save('animation2.gif', writer='imagemagick', fps=2)
particles_df[0].head(10)
filename = 'animation1.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
filename = 'animation2.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vx.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].px.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("x coordinate of particles");
axes[1].set_title("momentum of particle in x direction ");

f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vy.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].py.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("y coordinate of particles");
axes[1].set_title("momentum of particle in y direction ");

f, axes = plt.subplots(1, 2, figsize=(15,3));
sns.distplot(particles_df[0].vz.values, color='green', rug=False, ax=axes[0])
sns.distplot(particles_df[0].pz.values, color='green', rug=False, ax=axes[1])
axes[0].set_title("z coordinate of particles");
axes[1].set_title("momentum of particle in z direction ");

f, axes = plt.subplots(1, 2, figsize=(15,5));
sns.regplot(x=particles_df[0][:12000].vx.values, y=particles_df[0][:12000].vy.values, fit_reg=False, color='#ff4c64', ax=axes[0])
sns.regplot(x=particles_df[0][:12000].vx.values, y=particles_df[0][:12000].vz.values, fit_reg=False, color='#ff4c64', ax=axes[1])
axes[0].set_title("x and y position of particles");
axes[0].set(xlabel='x', ylabel='y');
axes[1].set_title("x and z position of particles");
axes[1].set(xlabel='x', ylabel='z');
plt.figure(figsize=(5, 4))
cnts = particles_df[0]['q'].value_counts()
pie(cnts.values, labels=cnts.index, colors=['#8ded82', '#f45342']);
show()
dist(particles_df[0], 'nhits', 10, 'red', 'particles.nhits distribution', kde=True)
psmall = particles_df[0][['vx','vy','vz']]
sns.pairplot(psmall, palette='husl', size=6)
plt.show()
truth_df[0].head(10)
dist(truth_df[0], 'weight', 10, 'red', 'truth.weight distribution', kde=True)