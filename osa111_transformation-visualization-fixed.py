####################################################################### IMPORT
import os
import numpy as np
import pandas as pd
from trackml.dataset import load_event
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA 
%matplotlib inline
####################################################################### TRANSFORMATION FUNCTIONS
# Convert to Cylindrical 
def cartesian_to_cylindrical(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    z = z
    return r, phi, z

# Convert to 3D Polar
def cartesian_to_3d_polar(x,y,z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    s  = np.sin(phi)
    c  = np.cos(phi)
    return r, s, c

# transform data to one of the above
def transform(data, transformation = "cylindrical"):
    data = data.copy()
    X = data.x.values
    Y = data.y.values
    Z = data.z.values
    if transformation == "cylindrical":
        data["x"], data["y"], data["z"] = cartesian_to_cylindrical(X,Y,Z)
        data.rename(columns={'x': 'r', 'y': 'phi', 'z': 'z'}, inplace=True)
    elif transformation == "polar":
        data["x"], data["y"], data["z"] = cartesian_to_3d_polar(X,Y,Z)
        data.rename(columns={'x': 'r', 'y': 's', 'z': 'c'}, inplace=True)
    
    x, y, z = None, None, None 
    return data


####################################################################### Getting Tracks 
def get_tracks(data, tracks_n=100,
               include_zero_weights=False,
               include_zero_ID = False, 
               coordinates = ['x','y','z'] ):
    # cordinates= normal or polar or cylindrical 
    data = data.copy()
    # remove zero weight particles (the one that is lower than 3 hits & and random noise)
    if include_zero_weights == False:
        data = data[data["weight"] > 0]
    if include_zero_ID == False:
        data = data[data["particle_id"] > 0]
    
    # get unique particle ids
    track = truth.particle_id.unique()
    
    if tracks_n > 0 and tracks_n < track.size:
        # select random particle ID (i.e., random tracks)
        # would work if tracks is a dataframe
        # tracks = tracks.loc[np.random.choice(tracks.index, size=n, replace = False)] 
        track = track[np.random.choice(track.shape[0], size=tracks_n, replace = False)]
    
    # get selected tracks only
    data = data[data["particle_id"].isin(track)]
    # replace particle ids by 1,2,3,4,5,6...n ... make iterating through tracks easier  
    data["particle_id"] = pd.factorize(data["particle_id"])[0]
    # Change     
    return data
    


####################################################################### Retrieval Functions 
def get_hits(train = 1, event = 1000, sample_n = 0):
    event_prefix = "event00000" + str(event)
    hits, _, _, _ = load_event(os.path.join('../input/train_1', event_prefix))
    _ = None
    if sample_n > 0 and sample_n < hits.shape[0]:
        hits = hits.sample(sample_n)
    return hits

def get_truth(train = 1, event = 1000, sample_n = 0):
    event_prefix = "event00000" + str(event)
    _, _, _, truth = load_event(os.path.join('../input/train_1', event_prefix))
    _ = None
    if sample_n > 0 and sample_n < truth.shape[0]:
        truth = truth.sample(sample_n)
    return truth



################## Constants
train_file = 1
event = 1000
sample_n = 80000
tracks_n = 50

####################################################################### Retrieve Data
# Get Hits
hits = get_hits(train = train_file, event = event, sample_n = sample_n)
hits_c = transform(hits, "cylindrical")
hits_p = transform(hits, "polar")
# Get Truths
truth = get_truth(train = train_file, event = event, sample_n = sample_n)
truth.rename(columns={'tx': 'x', 'ty': 'y', 'tz': 'z'}, inplace=True)
truth_c = transform(truth, "cylindrical")
truth_p = transform(truth, "polar")


########################################################################### PCA
### Apply PCA to Hits (needs cleaning up)
def doPCA(data):
    pca = PCA(n_components=3)
    pca.fit(data)
    return pca 

p = doPCA(hits[["x","y","z"]])
hits["x_"], hits["y_"] = p.transform(hits[["x","y","z"]])[:,0], p.transform(hits[["x","y","z"]])[:,1]
#print("hits normal score:" , p.explained_variance_ratio_)

p = doPCA(hits_c[["r","phi","z"]])
hits_c["x_"], hits_c["y_"] = p.transform(hits_c[["r","phi","z"]])[:,0], p.transform(hits_c[["r","phi","z"]])[:,1]
#print("hits cylind score:" , p.explained_variance_ratio_)
p = doPCA(hits_p[["r","s","c"]])
hits_p["x_"], hits_p["y_"] = p.transform(hits_p[["r","s","c"]])[:,0], p.transform(hits_p[["r","s","c"]])[:,1]
#print("hits polar score:" , p.explained_variance_ratio_)

### Apply PCA to Tracks 
p = doPCA(truth[["x","y","z"]])
truth["x_"], truth["y_"] = p.transform(truth[["x","y","z"]])[:,0], p.transform(truth[["x","y","z"]])[:,1]
#print("truth normal score:" , p.explained_variance_ratio_)

p = doPCA(truth_c[["r","phi","z"]])
truth_c["x_"], truth_c["y_"] = p.transform(truth_c[["r","phi","z"]])[:,0], p.transform(truth_c[["r","phi","z"]])[:,1]
#print("truth cylind score:" , p.explained_variance_ratio_)

p = doPCA(truth_p[["r","s","c"]])
truth_p["x_"], truth_p["y_"] = p.transform(truth_p[["r","s","c"]])[:,0], p.transform(truth_p[["r","s","c"]])[:,1]
#print("truth polar score:" , p.explained_variance_ratio_)


# Get Tracks (after PCA)
tracks = get_tracks(truth, tracks_n= tracks_n, coordinates = ['x','y','z'])
tracks_c = get_tracks(truth_c, tracks_n= tracks_n, coordinates = ['r','phi','z'])
tracks_p = get_tracks(truth_p, tracks_n= tracks_n, coordinates = ['r','s','c'])

##### PLOTTING Normal data
plt.figure(1, figsize=(10,10))
#plt.figure(figsize=(15,15))
ax1 = plt.axes(projection='3d')
ax1.scatter(hits.z, hits.y, hits.x, s=5, alpha=0.5)
ax1.set_xlabel('z (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_zlabel('x (mm)')
plt.title("normal x, y & z")
plt.show()

plt.figure(2, figsize=(10,10))
ax2 = plt.axes(projection='3d')
ax2.scatter(hits_c.z, hits_c.phi, hits_c.r, s=5, alpha=0.5)
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('phi')
ax2.set_zlabel('r')
plt.title("Cylindrical z, phi & r")
plt.show()

# 3D polar
plt.figure(3, figsize=(10,10))
ax2 = plt.axes(projection='3d')
ax2.scatter(hits_p.r, hits_p.s, hits_p.c, s=5, alpha=0.5)
ax2.set_xlabel('r')
ax2.set_ylabel('sin(theta)')
ax2.set_zlabel('cos(theta)')
plt.title(" 3D Polar r, s & c")
plt.show()

# Plotting Tracks 
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

for i in range(tracks["particle_id"].max()):
    t = tracks[tracks.particle_id == i]
    ax.plot3D(t.z, t.x, t.y)
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0)
ax.scatter(-3000,-3000,-3000, s=0)
plt.show()

plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

for i in range(tracks_c["particle_id"].max()):
    t = tracks_c[tracks_c.particle_id == i]
    ax.plot3D(t.z, t.phi, t.r)
ax.set_xlabel('z (mm)')
ax.set_ylabel('phi')
ax.set_zlabel('r')
# These two added to widen the 3D space
plt.show()


plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')

for i in range(tracks_p["particle_id"].max()):
    t = tracks_p[tracks_p.particle_id == i]
    ax.plot3D(t.r, t.s, t.c)
ax.set_xlabel('r')
ax.set_ylabel('s')
ax.set_zlabel('c')
plt.show()
##### PLOT 2D x,y & phi,r

plt.figure(4, figsize=(15,15))
plt.subplot(321)
plt.scatter(hits.y, hits.x, s=5, alpha=0.5)
plt.xlabel('y (mm)')
plt.ylabel('x (mm)')

plt.subplot(322)
plt.scatter(hits.y, hits.z, s=5, alpha=0.5)
plt.xlabel('y')
plt.ylabel('z')
# x & z looks same as y & z

plt.subplot(323)
plt.scatter(hits_c.phi, hits_c.r, s=5, alpha=0.5)
plt.xlabel('phi')
plt.ylabel('r')
# r & z looks same as phi & r

plt.subplot(324)
plt.scatter(hits_c.phi, hits_c.z, s=5, alpha=0.5)
plt.xlabel('phi')
plt.ylabel('z')

plt.subplot(325)
plt.scatter(hits_p.r, hits_p.s, s=5, alpha=0.5)
plt.xlabel('r')
plt.ylabel('s')
# r & c looks same as r & s

plt.subplot(326)
plt.scatter(hits_p.s, hits_p.c, s=5, alpha=0.5)
plt.xlabel('s')
plt.ylabel('c')
plt.show()
### Plot Graphs above with Tracks, each unique color is a track
plt.figure(4, figsize=(20,20))
plt.subplot(321)
plt.scatter(hits.y, hits.x, s=5, alpha=0.5)
for i in range(tracks["particle_id"].max()):
    t = tracks[tracks.particle_id == i]
    plt.scatter(t.y, t.x)
plt.xlabel('y (mm)')
plt.ylabel('x (mm)')

plt.subplot(322)
plt.scatter(hits.y, hits.z, s=5, alpha=0.5)
for i in range(tracks["particle_id"].max()):
    t = tracks[tracks.particle_id == i]
    plt.scatter(t.y, t.z)
plt.xlabel('y')
plt.ylabel('z')
# x & z looks same as y & z

plt.subplot(323)
plt.scatter(hits_c.phi, hits_c.r, s=5, alpha=0.5)
for i in range(tracks_c["particle_id"].max()):
    t = tracks_c[tracks_c.particle_id == i]
    plt.scatter(t.phi, t.r)
plt.xlabel('phi')
plt.ylabel('r')
# r & z looks same as phi & r

plt.subplot(324)
plt.scatter(hits_c.phi, hits_c.z, s=5, alpha=0.5)
for i in range(tracks_c["particle_id"].max()):
    t = tracks_c[tracks_c.particle_id == i]
    plt.scatter(t.phi, t.z)
plt.xlabel('phi')
plt.ylabel('z')

plt.subplot(325)
plt.scatter(hits_p.r, hits_p.s, s=5, alpha=0.5)
for i in range(tracks_p["particle_id"].max()):
    t = tracks_p[tracks_p.particle_id == i]
    plt.scatter(t.r, t.s)
plt.xlabel('r')
plt.ylabel('s')
# r & c looks same as r & s

plt.subplot(326)
plt.scatter(hits_p.s, hits_p.c, s=5, alpha=0.5)
for i in range(tracks_p["particle_id"].max()):
    t = tracks_p[tracks_p.particle_id == i]
    plt.scatter(t.s, t.c)
plt.xlabel('s')
plt.ylabel('c')
plt.show()
## Plot hits after dimensionallity reduction 
plt.figure(5, figsize=(10,10))
plt.scatter(hits.x_, hits.y_, s=5, alpha=0.5)
plt.title("Cartesian reduced")
plt.show()


plt.figure(6, figsize=(10,10))
plt.scatter(hits_c.x_, hits_c.y_, s=5, alpha=0.5)

plt.title("Cylindrical Reduced")
plt.show()

plt.figure(7, figsize=(10,10))
plt.scatter(hits_p.x_, hits_p.y_, s=5, alpha=0.5)

plt.title("Polar reduced")
plt.show()

# Plot graphs above with tracks after PCA applied to both
plt.figure(7, figsize=(10,10))
plt.scatter(hits.x_, hits.y_, s=5, alpha=0.5)

for i in range(tracks["particle_id"].max()):
    t = tracks[tracks.particle_id == i]#.sort_values("y2")
    plt.plot(t.x_, t.y_)

plt.title("Reduced Tracks & Hits (cartesian)")
plt.show()

#tracks.sort_values("y2")
plt.figure(8, figsize=(10,10))
plt.scatter(hits_c.x_, hits_c.y_, s=5, alpha=0.5)

for i in range(tracks_c["particle_id"].max()):
    t = tracks_c[tracks_c.particle_id == i]#.sort_values("y2")
    plt.scatter(t.x_, t.y_)

plt.title("Reduced Tracks & Hits (cylindrical)")
plt.show()

plt.figure(9, figsize=(10,10))
plt.scatter(hits_p.x_, hits_p.y_, s=5, alpha=0.5)
for i in range(tracks_p["particle_id"].max()):
    t = tracks_p[tracks_p.particle_id == i]#.sort_values("y2")
    plt.scatter(t.x_, t.y_)
plt.title("Reduced Tracks & Hits (polar)")
plt.show()
