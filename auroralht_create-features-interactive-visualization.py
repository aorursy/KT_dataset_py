import pdb
import os
import copy

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
%matplotlib inline

#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
device = pd.read_csv('../input/detectors.csv')
event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))

mem_bytes = (hits.memory_usage(index=True).sum() 
             + cells.memory_usage(index=True).sum() 
             + particles.memory_usage(index=True).sum() 
             + truth.memory_usage(index=True).sum())
print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))
truth.sort_values(by="hit_id",inplace=True)
truth.head()
truth['hasweight'] = ~np.equal( truth.weight.values, 0 )
truth[['particle_id','weight','hasweight']].head()
# store the magnitude of the momentum
truth['tp'] = np.sqrt(truth['tpx']**2+truth['tpy']**2+truth['tpz']**2)
truth.head()
particles.head()
# add hit_id to help the merge in future
particles['hit_id'] = -1
particles['p'] = np.sqrt(particles['px']**2+particles['py']**2+particles['pz']**2)
particles.head()
init_truth = particles.rename({
    'vx' : 'tx',
    'vy' : 'ty',
    'vz' : 'tz',
    'px' : 'tpx',
    'py' : 'tpy',
    'pz' : 'tpz',
    'p'  : 'tp'
}, axis=1)
init_truth.drop('nhits', axis=1, inplace=True)
# the number of particles in the particles and truth
inituni_par = init_truth.particle_id.unique()
uni_par = truth.particle_id.unique()
len(inituni_par), len(uni_par)
# how many particles do they share
inter = np.intersect1d(uni_par, inituni_par)
len(inter)
# what is the one can't find in the particles
np.setdiff1d(uni_par, inter)
weight_map = truth.groupby('particle_id').first()['weight']
charge_map = init_truth.groupby('particle_id').first()['q']
truth['q'] = truth.particle_id.map(charge_map)
truth.fillna(0, inplace=True)
init_truth['weight'] = init_truth.particle_id.map(weight_map)
init_truth.fillna(0, inplace=True)
init_truth['hasweight'] = ~np.equal(init_truth.weight, 0)
truth.set_index('hit_id',inplace=True)
init_truth.set_index('hit_id',inplace=True)
fulltruth = init_truth.append(truth, sort=True)
fulltruth.head()
fulltruth['R'] =  np.sqrt( fulltruth['tpx']**2 + fulltruth['tpy']**2 ) / fulltruth['q']
tgroups = truth.groupby("hasweight")

unrel = tgroups.get_group(False)
unrel_ = unrel.groupby('particle_id').first()

rel = tgroups.get_group(True)
rel_ = rel.groupby('particle_id').first()
fig, axs = plt.subplots(1,2, figsize=(18,6))
axs[0].set_title('zero weight')
sns.distplot(unrel.tp, hist=True, kde=False, ax = axs[0] )
axs[1].set_title('has weight')
sns.distplot(rel.tp, hist=True, kde=False, ax= axs[1])
par0 = truth[truth.particle_id==0]
len(truth), len(par0), len(par0)/len(truth)    
par0.tp.describe()
stdlayout3d = dict(
    width=800,
    height=700,
        
    autosize=False,
    title= 'unknown',
    scene=dict(
        xaxis=dict(
            title = "unknown x",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title = "unknown y",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            title = "unknown z",
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-1.7428, y=1.0707, z=0.7100,)
        ),
        aspectratio = dict(x=1, y=1, z=0.7),
        aspectmode = 'manual'
    ),
)

def layout_costom3d(xtitle, ytitle, ztitle, title, xrange=None, yrange=None, zrange=None):
    layout = copy.deepcopy(stdlayout3d)
    layout['scene']['xaxis']['title'] = xtitle 
    layout['scene']['yaxis']['title'] = ytitle
    layout['scene']['zaxis']['title'] = ztitle
    if xrange is not None: layout['scene']['xaxis']['range'] = xrange
    if yrange is not None: layout['scene']['yaxis']['range'] = yrange
    if zrange is not None: layout['scene']['zaxis']['range'] = zrange
    layout['title'] = title
    return layout
stdlayout2d = dict(
    height = 800,
    width = 800,
    title = 'unknown',
    yaxis = dict(),
        #zeroline = False,

    xaxis = dict(),
        #zeroline = False,
)    
def layout_costom2d(xtitle, ytitle, title, xrange=None, yrange=None, zrange=None):
    layout = copy.deepcopy(stdlayout2d)
    layout['xaxis']['title'] = xtitle 
    layout['yaxis']['title'] = ytitle    
    if xrange is not None: layout['xaxis']['range'] = xrange
    if yrange is not None: layout['yaxis']['range'] = yrange
    layout['title'] = title
    return layout   

xyzlayout = layout_costom3d('z axis(mm)', 'x axis(mm)', 'y axis(mm)', 'sample trajectories')
xylayout = layout_costom2d('x axis(mm)', 'y axis(mm)', 'sample trajectories', [-1000, 1000], [-1000, 1000])
def set_marker(pp, pp_name, isnoise, ms, cmin, cmax):
    marker = dict(
        size=ms,
        symbol= "square" if isnoise else "circle",
    )
    #pdb.set_trace()
    if pp is not None:
        marker['color'] = pp
        marker['colorscale']='Rainbow'
        marker['colorbar']=dict(
                title = pp_name,
                x = 1.20
            )
        marker['cmin'] = cmin
        marker['cmax'] = cmax
    return marker

def set_line( width=1):
    line = dict(
        width=1
    )
def plotly_3d(x, y, z, pp=None, pp_name=None, isnoise=False, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):     
    marker = set_marker(pp, pp_name, isnoise, ms, cmin, cmax)        
    trace = go.Scatter3d(
        mode=mode,
        visible=visible,
        x=x, y=y, z=z,
        marker= marker,
        line = dict(width=1)
    )
    return [trace]

def plot_df(df, xyzcols, n=10, pproperty=None, pids=None, visible="legendonly", mode=None, ms=2, cmin=0, cmax=1):
    particlegroup = df.groupby('particle_id')
    if pids is None:
        pids = np.random.choice(df.particle_id.unique(), n)
    
    particles = [particlegroup.get_group(pid) for pid in pids]
    
    data = []
    xc, yc, zc = xyzcols
    for particle in particles:
        trace=plotly_3d(
            x=particle[xc],
            y=particle[yc],
            z=particle[zc],
            pp=particle[pproperty] if pproperty is not None else None,
            pp_name=pproperty,
            isnoise=particle.weight.values[0] == 0,
            visible=visible,
            mode=mode,
            ms=ms, cmin=cmin, cmax=cmax
        )
        data+=trace
    return data
def plotly_2d(x, y, pp=None, pp_name=None, isnoise=False, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):
        
    marker = set_marker(pp, pp_name, isnoise, ms, cmin, cmax)       
    trace = go.Scatter(
        x = x,
        y = y,
        mode = mode,
        marker = marker,
        line = dict(width=1)
    )
    return [trace]
        
def plot_df2d(df, xycols, n=10, pproperty=None, pids=None, visible="legendonly", mode=None, ms=4, cmin=0, cmax=1):
    particlegroup = df.groupby('particle_id')
    if pids is None:
        pids = np.random.choice(df.particle_id.unique(), n)
    
    particles = [particlegroup.get_group(pid) for pid in pids]
        
    data = []
    xc, yc = xycols
    for particle in particles:
        trace=plotly_2d(
            x=particle[xc],
            y=particle[yc],
            pp=particle[pproperty] if pproperty is not None else None,
            pp_name=pproperty,
            isnoise=particle.weight.values[0] == 0,
            visible=visible,
            mode=mode,
            ms=ms, cmin=cmin, cmax=cmax
        )
        data+=trace
    return data 
data = plotly_3d(particles.vz, particles.vx, particles.vy, particles.p, 'momentum', visible=True, ms=1, mode='markers')
layout = xyzlayout.copy()
layout['title'] = 'initial position'
iplot(dict(data=data, layout=layout), filename='local')
data = plot_df(truth, ['tz','tx','ty'], n=10, visible=True)
iplot(dict(data=data, layout=xyzlayout))
data = plot_df(truth, ['tz','tx','ty'], pids=[0], visible=True, mode='markers', ms=1)
layout = xyzlayout.copy()
layout['title'] = 'Noise'
iplot(dict(data=data, layout=layout))
data = plot_df2d(truth, ['tx', 'ty'], n=100, mode=None)
iplot(dict(data=data, layout=xylayout))
hits.head()
cells.head()
cells_ = cells.set_index('hit_id')
cells_.drop(['ch0','ch1'], axis=1, inplace=True)
cells_ = cells_.groupby('hit_id').agg('sum')
cells_.head()
hits_ = hits.set_index('hit_id')
info = cells_.join(hits_)
cheat = info.join(fulltruth)
cheat.dropna(axis=0)
cheat.R.replace(np.inf, 10000, inplace=True)
cheat.head()
pids = cheat.particle_id.unique()
samples = np.random.choice(pids, 2)
for sample in samples:
    if sample != 0:   
        ax = sns.distplot( cheat[cheat.particle_id==sample].value , kde=False, bins=np.linspace(0,1,10) )
ax.set_xlim([0,1])
fig, axs =plt.subplots(1,3,figsize=(18,4))

ax = sns.distplot(cheat.value, ax=axs[0], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('overall')
ax = sns.distplot(cheat[cheat.particle_id == 0].value, ax=axs[1], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('noise')
ax = sns.distplot(cheat[~(cheat.particle_id == 0)].value, ax=axs[2], kde=False, bins=np.linspace(0,1,20))
ax.set_xlim([0,1])
ax.set_title('not noise');
data = plot_df(cheat, ['tx', 'ty', 'tz'], n=30 ,pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))
highsig = cheat.value > 0.5
highsigpart = cheat.particle_id[highsig].unique()
lowsigpart = np.setdiff1d(cheat.particle_id,highsigpart)
cheat.particle_id.nunique(), len(highsigpart), len(lowsigpart)
data = plot_df(cheat, ['tx', 'ty', 'tz'], pids=np.random.choice(highsigpart, 10), pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))
data = plot_df(cheat, ['tx', 'ty', 'tz'], pids=np.random.choice(lowsigpart, 10), pproperty='value', visible=True)
iplot(dict(data=data, layout=xyzlayout))
data = plot_df(cheat, ['tx', 'ty', 'tz'], pproperty='tp', visible=True, cmin=0, cmax=2, ms=2)
iplot(dict(data=data, layout=xyzlayout))
data = plot_df(cheat, ['tx', 'ty', 'tz'], pproperty='q', visible=True, cmin=-1, cmax=1, ms=4)
iplot(dict(data=data, layout=xyzlayout))
data = plot_df2d(cheat, ['tx', 'ty'], pproperty='q', visible=True, cmin=-1, cmax=1, ms=6, mode='markers')
iplot(dict(data=data, layout=xylayout))
data = plot_df2d(cheat, ['tx', 'ty'], pproperty='R', visible=True, cmin=-1, cmax=1, ms=6, mode='markers')
iplot(dict(data=data, layout=xylayout))
def xyz2c(df, cols, newcols):
    x, y, z = df[cols[0]], df[cols[1]], df[cols[2]]
    #pdb.set_trace()
    cr, cpsi, cz = newcols
    r = np.sqrt( x**2 + y**2 )
    psi = np.arctan(y/x)
    df[cr] = r
    df[cpsi] = psi
    df[cz] = z
def xyz2s(df, cols, newcols):
    x, y, z = df[cols[0]], df[cols[1]], df[cols[2]]
    cr, cpsi, ctheta = newcols
    r = np.sqrt( x**2 + y**2 + z**2 )
    psi = np.arctan(y/x)
    theta = np.arccos(z/r)
    df[cr] = r
    df[cpsi] = psi
    df[ctheta] = theta
xyz2c(cheat, ['tx','ty','tz'], ['trc','tpsic','tzc'])
xyz2s(cheat, ['tx','ty','tz'], ['trs','tpsis','tthetas'])
## Plotting the trajectory in new coordinates
### in Spherical Coordinates
spherelayout = layout_costom3d('theta (radians)','radius (mm)', 'psi (radians)', 'sample trajectories in spherical coor',xrange=[0, np.pi], zrange=[-np.pi, np.pi])
data = plot_df(cheat, ['tthetas', 'trs', 'tpsis'], visible=True)
iplot(dict(data=data, layout=spherelayout))
### in Cylindricall Coordinates
cylindricallayout = layout_costom3d('z (mm)','radius (mm)', 'psi (radians)', 'sample trajectories in cylindrical coor', zrange=[-np.pi, np.pi])
data = plot_df(cheat, ['tz', 'trc', 'tpsic'], visible=True)
iplot(dict(data=data, layout=cylindricallayout))
xyz2c(device, ['cx','cy','cz'], ['rc','psic','zc'])
xyz2s(device, ['cx','cy','cz'], ['rs','psis','thetas'])
zrlayout = layout_costom2d('z (mm)','radius (mm)', 'sample trajectories on rz plane')
data = plotly_2d(x=device.zc, y=device.rc, isnoise=True, ms=5, mode='markers')
data += plot_df2d(cheat, ['tzc','trc'])
iplot(dict(data=data, layout=zrlayout))