import os
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import trackml
from trackml.dataset import load_event
cFirstEvent=1010
cEventDataDir='../input/train_1'
def getPath(pDataDir,pEventID) : 
    return '%s/event%09d' % (pDataDir, pEventID)


hits, cells, particles, truth = load_event(getPath(cEventDataDir,cFirstEvent))
particles.head()
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import math as math
# Load in convex hull method
from scipy.stats.stats import pearsonr
from scipy.spatial import ConvexHull
#circle
from scipy import optimize

nMinHits=5
#draw track + hits 
def getTrackParameters(pIndex) : 
    dataFrame = pd.DataFrame(particles)
    
# start with those that have 5 hits 
def getTracks(sampleSize) : 
    dataFrame = pd.DataFrame(particles)
    dataFrame = dataFrame[dataFrame['nhits']>=nMinHits]
    # get unique list of particle IDs 
    particle_IDs = np.random.choice(dataFrame.particle_id.unique(),sampleSize)
    print(particle_IDs)
    dataFrame = pd.DataFrame(truth)
    df_truth = dataFrame[dataFrame['particle_id'].isin(particle_IDs)]
    return df_truth

def getHitsFromTracks(df_truth, sampleSize) : 
    dataFrame = pd.DataFrame(hits)
    df_hits = dataFrame[dataFrame['hit_id'].isin(df_truth.hit_id)]
    return df_hits

def getOtherHits(df_truth, sampleSize) : 
    dataFrame = pd.DataFrame(hits)
    df_hits = dataFrame[dataFrame['hit_id'].isin(df_truth.hit_id)== False]
    return  df_hits.sample(n=sampleSize)

#return truths for a given particle 
def getTruth(pTruths, particleID) :
    dataFrame = pd.DataFrame(pTruths)
    df_t = dataFrame[dataFrame['particle_id'] == particleID]
    return df_t


#return hits in a given volume 
def getHitsForVolume(pHits, pVolumeID) : 
    dataFrame = pd.DataFrame(pHits)
    df_v = dataFrame[dataFrame['volume_id'] == pVolumeID]
    #df_v = df_v[df_v['layer_id'] < 6]
    return df_v

#return hits in a given volume 
def getHitsForVolume_perLayer(pHits, pVolumeID, pLayerID) : 
    dataFrame = pd.DataFrame(pHits)
    df_v = dataFrame[dataFrame['volume_id'] == pVolumeID]
    df_v = df_v[df_v['layer_id'] == pLayerID]
    return df_v

# make things look familiar...
#plots hits in (x,y) [cartesian] and (z,r) coordinate system [cylindrical]
def showHitsForVolume(pHits, pVolumeID) : 
    df_v = getHitsForVolume(pHits,pVolumeID)   
    #now estimate r-coordinate (in x,y plane)
    r = (df_v.x**2 + df_v.y**2)**0.5
    phi = np.arctan(df_v.y/df_v.x)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(df_v.x,df_v.y, 'bs')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')

    plt.subplot(122)
    plt.plot(df_v.z,r, 'bs')
    plt.xlabel('z [cm]')
    plt.ylabel('r [cm]')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=1.55, hspace=0.25, wspace=0.35)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=1.55, hspace=0.25, wspace=0.35)
    
    return plt

def showHitsForVolume_perLayer(pHits, pVolumeID, pLayerID) : 
    df_v = getHitsForVolume_perLayer(pHits,pVolumeID,pLayerID)   
    #now estimate r-coordinate (in x,y plane)
    r = (df_v.x**2 + df_v.y**2)**0.5
    phi = np.arctan(df_v.y/df_v.x)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(df_v.x,df_v.y, 'bs')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')

    plt.subplot(122)
    plt.plot(df_v.z,r, 'bs')
    plt.xlabel('z [cm]')
    plt.ylabel('r [cm]')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=1.55, hspace=0.25, wspace=0.35)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=1.55, hspace=0.25, wspace=0.35)
    
    return plt

def showHitsForParticle(pTruth,particleID) : 
    df_t = getTruth(pTruth,particleID)
    r = (df_t.tx**2 + df_t.ty**2)**0.5
    plt.figure(1)
    plt.subplot(121)
    plt.plot(df_t.tx,df_t.ty, 'bs')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    
    plt.subplot(122)
    plt.plot(df_t.tz,r, 'bs')
    plt.xlabel('z [cm]')
    plt.ylabel('r [cm]')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=1.55, hspace=0.25, wspace=0.35)    
    return plt

def draw(x,y) : 
    plt.figure(1)
    plt.plot(x,y, 'bs')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    
    return plt 

nTrueTracks=1
nFakeHits=5
dh = pd.DataFrame(hits)
dh = dh[np.fabs(dh['z']) < 1]
d_t = getTracks(nTrueTracks)
d_ht = getHitsFromTracks(d_t,nTrueTracks)
d_hf = getOtherHits(d_t,nFakeHits)
r_ht = np.sqrt(d_ht.x**2 + d_ht.y**2)
d_ht['r'] = r_ht
r_hf = np.sqrt(d_hf.x**2 + d_hf.y**2)
d_hf['r'] = r_hf
d = pd.concat([d_ht, d_hf])
plt.plot(dh.x,dh.y,'or')
wdir = os.getcwd()

hits_cols = "hit_id,x,y,z,volume_id,layer_id,module_id,event_name"
particle_cols = "particle_id,vx,vy,vz,px,py,pz,q,nhits,event_name"
truth_cols = "hit_id,particle_id,tx,ty,tz,tpx,tpy,tpz,weight,event_name"
cells_cols = "hit_id,ch0,ch1,value,event_name"

hits_df = pd.DataFrame(columns = hits_cols.split(","))
particle_df = pd.DataFrame(columns=particle_cols.split(","))
truth_df =  pd.DataFrame(columns = truth_cols.split(","))
cells_df = pd.DataFrame(columns= cells_cols.split(','))
def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(*c)
    return Ri - Ri.mean()

x = d['x']
y = d['y']
x_m = np.mean(x)
y_m = np.mean(y)

center_estimate = x_m,y_m
center_2, ier = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2       = calc_R(*center_2)
R_2        = Ri_2.mean()
residu_2   = sum((Ri_2 - R_2)**2)
print(center_2)
xC = np.linspace((np.min(x)-0.1*R_2), (np.max(x)+0.1*R_2), 100)
yC = np.linspace((np.min(y)-0.1*R_2), (np.max(y)+0.1*R_2), 100)
X, Y = np.meshgrid(xC,yC)
F = (X-xc_2)**2 + (Y-yc_2)**2 - R_2**2
plt.plot(x, y, 'ok')
plt.show()
hits, cells, particles, truth = load_event(getPath(cEventDataDir,cFirstEvent))
hits.head()
hits.describe()
cells.head()
cells.describe()
particles[(particles['q'] != -1) & (particles['q'] != 1)]
truth.head()
truth.describe()
track = truth[truth['particle_id'] == 4503737066323968]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for hit_id, hit in track.iterrows():
    ax.scatter(hit.tx, hit.ty, hit.tz)
def calc_curvature(data_fr):
    x = data_fr.tx
    y = data_fr.ty
    z = data_fr.tz
    ddx  = np.diff(np.diff(x))
    ddy  = np.diff(np.diff(y))
    ddz  = np.diff(np.diff(z))
#     take the mean curvature (not the sum) to avoid bias 
#     since some particles generate more hits and others less
    return np.sqrt(ddx**2 + ddy**2 + ddz**2).mean() 
df  = pd.merge(hits_df,truth_df,how = 'left', on = ['hit_id','event_name'])
df = df[df['particle_id']!= 0] # drop particle 0 
grouped = df.groupby(['event_name','particle_id'])
curvatures = grouped.apply(calc_curvature)
import seaborn as sns
g = sns.jointplot(hits.x, hits.y,  s=1, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.x, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()
g = sns.jointplot(hits.z, hits.y, s=1, size=12)
g.ax_joint.cla()
plt.sca(g.ax_joint)

volumes = hits.volume_id.unique()
for volume in volumes:
    v = hits[hits.volume_id == volume]
    plt.scatter(v.z, v.y, s=3, label='volume {}'.format(volume))

plt.xlabel('Z (mm)')
plt.ylabel('Y (mm)')
plt.legend()
plt.show()
hits_sample = hits.sample(8000)
sns.pairplot(hits_sample, hue='volume_id', size=8)
plt.show()