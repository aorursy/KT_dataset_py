#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

path_to_train = "../input/train_1"
event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))


def get_training_sample(path_to_data, event_names):
    events = []
    track_id = 0

    for name in event_names:
        # Read an event
        hits, cells, particles, truth = load_event(os.path.join(path_to_data, name))

        # Generate new vector of particle id
        particle_ids = truth.particle_id.values
        particle2track = {}
        for pid in np.unique(particle_ids):
            particle2track[pid] = track_id
            track_id += 1
        hits['particle_id'] = [particle2track[pid] for pid in particle_ids]

        # Collect hits
        events.append(hits)

    # Put all hits into one sample with unique tracj ids
    data = pd.concat(events, axis=0)
    return data


start_event_id = 1000
n_train_samples = 5
train_event_names = ["event0000{:05d}".format(i) for i in range(start_event_id, start_event_id+n_train_samples)]
train_data = get_training_sample(path_to_train, train_event_names)


class KNNScaledClusterer(object):
    def __init__(self):
        self.classifier = None

    def _preprocess(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values
        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        return X

    def fit(self, hits):
        X = self._preprocess(hits)
        y = hits.particle_id.values
        self.classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        self.classifier.fit(X, y)

    def predict(self, hits):
        X = self._preprocess(hits)
        labels = self.classifier.predict(X)
        return labels


model = KNNScaledClusterer()
model.fit(train_data)

path_to_event = os.path.join(path_to_train, "event0000{:05d}".format(start_event_id + n_train_samples + 1))
hits, cells, particles, truth = load_event(path_to_event)

labels = model.predict(hits)

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Std KNN score: ", score)

class KNNPolarClusterer(object):
    def __init__(self):
        self.classifier = None

    def _preprocess(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values
        r = np.sqrt(x**2 + y**2)
        t = np.arctan2(y, x)
        hits['r'] = r
        hits['t'] = t

        ss = StandardScaler()
        X = ss.fit_transform(hits[['r', 't', 'z']].values)
        return X

    def fit(self, hits):
        X = self._preprocess(hits)
        y = hits.particle_id.values
        self.classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        self.classifier.fit(X, y)

    def predict(self, hits):
        X = self._preprocess(hits)
        labels = self.classifier.predict(X)
        return labels
    
model = KNNPolarClusterer()
model.fit(train_data)

hits, cells, particles, truth = load_event(path_to_event)
labels = model.predict(hits)

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Polar KNN score: ", score)

from mpl_toolkits.mplot3d import Axes3D

def cart2spherical(cart):
    r = np.linalg.norm(cart, axis=0)
    theta = np.degrees(np.arccos(cart[2] / r))
    phi = np.degrees(np.arctan2(cart[1], cart[0]))
    return np.vstack((r, theta, phi))



NUM_PARTICLES = 100
truth_dedup = truth.drop_duplicates('particle_id')
truth_sort = truth_dedup.sort_values('weight', ascending=False)
truth_head = truth_sort.head(NUM_PARTICLES)

# Get points where the same particle intersected subsequent layers of the observation material
p_traj_list = []
for _, tr in truth_head.iterrows():
    p_traj = truth[truth.particle_id == tr.particle_id][['tx', 'ty', 'tz']]
    # Add initial position.
    #p_traj = (p_traj
    #          .append({'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}, ignore_index=True)
    #          .sort_values(by='tz'))
    p_traj_list.append(p_traj)
    
# Convert to spherical coordinate.
rtp_list = []
for p_traj in p_traj_list:
    xyz = p_traj.loc[:, ['tx', 'ty', 'tz']].values.transpose()
    rtp = cart2spherical(xyz).transpose()
    rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
    rtp_list.append(rtp_df)

# Plot with Cartesian coordinates.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for p_traj in p_traj_list:
    ax.plot(
        xs=p_traj.tx,
        ys=p_traj.ty,
        zs=p_traj.tz,
        marker='o')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm) -- Detection layers')
plt.title('Trajectories of top weights particles in Cartesian coordinates.')

# Plot with spherical coordinates.
fig2 = plt.figure(figsize=(10, 10))
ax = fig2.add_subplot(111, projection='3d')
for rtp_df in rtp_list:
    ax.plot(
        xs=rtp_df.theta,
        ys=rtp_df.phi,
        zs=rtp_df.r,
        marker='o')
ax.set_xlabel('Theta (deg)')
ax.set_ylabel('Phi (deg)')
ax.set_zlabel('R  (mm) -- Detection layers')
plt.title('Trajectories of top weights particles in spherical coordinates.')
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

class DBSCANClusterer(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    
    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        
        return X
    
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=1, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
        return labels
    
model = DBSCANClusterer(eps=0.008)
labels = model.predict(hits)
submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Your score: ", score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

class DBSCANPolarClusterer(object):
    
    def __init__(self, eps):
        self.eps = eps
        
    
    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        t = np.arctan2(y, x)
        p = np.arccos(y / r)
        hits['r'] = r
        hits['t'] = t
        hits['p'] = p

        ss = StandardScaler()
        X = ss.fit_transform(hits[['r', 't', 'p']].values)
        X = ss.fit_transform(hits[['t', 'p']].values)
        
        return X
    
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=1, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
        return labels
    
model = DBSCANPolarClusterer(eps=0.008)
labels = model.predict(hits)
submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Your score: ", score)
path_to_train = "../input/train_1"
event_prefix = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

data = pd.merge(hits, truth[['hit_id', 'particle_id']], on='hit_id')

data.head()
data['r'] = np.sqrt(data.x*data.x + data.y*data.y + data.z*data.z)
#data.groupby('particle_id').transform(lambda x: print(x))
refs = []
for pid, dat in data.groupby('particle_id'):
    #print(dat[(dat.r - dat.r.median()) == (dat.r - dat.r.median()).min()])
    ref = dat[(dat.r - dat.r.median()) == (dat.r - dat.r.median()).min()].iloc[0]
    refs.append({'particle_id': pid, 'ref_x': ref.x, 'ref_y': ref.y, 'ref_z': ref.z})
    
refs = pd.DataFrame(refs)
data = pd.merge(data, refs, on='particle_id')
data[data.x == data.ref_x].head()
data.shape
# want to ignore particle_id == 0 because they aren't actually a single track and will fuck up the regression
datanonzero = data[data.particle_id != 0]
datanonzero.shape
# https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py

import torch
import torch.optim as optim

device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 3, 100, 3

# Create random Tensors to hold inputs and outputs
x = np.asarray(datanonzero[['x', 'y', 'z']])
y = np.asarray(datanonzero[['ref_x', 'ref_y', 'ref_z']])
from sklearn.preprocessing import StandardScaler
xs = StandardScaler()
ys = StandardScaler()
x = xs.fit_transform(x)
y = ys.fit_transform(y)
x = torch.autograd.Variable(torch.from_numpy(x)).float()
y = torch.autograd.Variable(torch.from_numpy(y)).float()

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# After constructing the model we use the .to() method to move it to the
# desired device.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Dropout(),
          torch.nn.Linear(H, D_out),
        ).to(device)
#model = model.train()

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4
learning_rate = 1e-6
weight_decay = 0.01
opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for t in range(500):
    model.zero_grad()
    opt.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    loss.backward()
    opt.step()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its data and gradients like we did before.
    #with torch.no_grad():
    #for param in model.parameters():
    #        param.data -= learning_rate * param.grad
    #        print(param.grad)
scaled = xs.transform(data[['x', 'y', 'z']])
data['sx'], data['sy'], data['sz'] = scaled.T

# polar
data['polar_r'] = np.linalg.norm(data[['x', 'y', 'z']], axis=1)
data['polar_p'] = np.degrees(np.arccos(data['z'] / data['polar_r']))
data['polar_t'] = np.degrees(np.arctan2(data['y'], data['x']))

ss_polar = StandardScaler()
polar_scaled = ss_polar.fit_transform(data[['polar_r', 'polar_p', 'polar_t']])
data['ss_polar_r'], data['ss_polar_p'], data['ss_polar_t'] = polar_scaled.T

# cylindrical
data['cylindrical_r'] = np.linalg.norm(data[['x', 'y']], axis=1)
data['cylindrical_t'] = np.degrees(np.arctan2(data['y'], data['x']))
data['cylindrical_z'] = data['z']

ss_cylindrical = StandardScaler()
cylindrical_scaled = ss_cylindrical.fit_transform(data[['cylindrical_r', 'cylindrical_t', 'cylindrical_z']])
data['ss_cylindrical_r'], data['ss_cylindrical_t'], data['ss_cylindrical_z'] = cylindrical_scaled.T

# radial norm
data['radn_x'] = data['x'] / np.linalg.norm(data[['x', 'y', 'z']], axis=1)
data['radn_y'] = data['y'] / np.linalg.norm(data[['x', 'y', 'z']], axis=1)
data['radn_z'] = data['z'] / np.linalg.norm(data[['x', 'y']], axis=1)

ss_radn = StandardScaler()
radn_scaled = ss_radn.fit_transform(data[['radn_x', 'radn_y', 'radn_z']])
data['ss_radn_x'], data['ss_radn_y'], data['ss_radn_z'] = radn_scaled.T

data['nn_med_x'], data['nn_med_y'], data['nn_med_z'] = model(torch.from_numpy(scaled).float()).data.numpy().T
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for (pid, dat), _ in zip(data.groupby('particle_id'), range(50)):
    if pid == 0: continue
    ax.plot(
        xs=dat.sx,
        ys=dat.sy,
        zs=dat.sz,
        marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('scaled space')
plt.show()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for (pid, dat), _ in zip(data.groupby('particle_id'), range(50)):
    if pid == 0: continue
    ax.plot(
        xs=dat.ss_polar_t,
        ys=dat.ss_polar_p,
        zs=dat.ss_polar_r,
        marker='o')
ax.set_xlabel('t')
ax.set_ylabel('p')
ax.set_zlabel('r')
plt.title('scaled polar space')
plt.show()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for (pid, dat), _ in zip(data.groupby('particle_id'), range(50)):
    if pid == 0: continue
    ax.plot(
        xs=dat.ss_cylindrical_t,
        ys=dat.ss_cylindrical_z,
        zs=dat.ss_cylindrical_r,
        marker='o')
ax.set_xlabel('t')
ax.set_ylabel('z')
ax.set_zlabel('r')
plt.title('scaled cylindrical space')
plt.show()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for (pid, dat), _ in zip(data.groupby('particle_id'), range(50)):
    if pid == 0: continue
    ax.plot(
        xs=dat.ss_radn_x,
        ys=dat.ss_radn_z,
        zs=dat.ss_radn_y,
        marker='o')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
plt.title('scaled radial norm space')
plt.show()
scaled = xs.transform(data[['x', 'y', 'z']])
data['sx'], data['sy'], data['sz'] = scaled.T

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
for (pid, dat), _ in zip(data.groupby('particle_id'), range(50)):
    if pid == 0: continue
    ax.plot(
        xs=dat.nn_med_x,
        ys=dat.nn_med_y,
        zs=dat.nn_med_z,
        marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Trajectories of top weights particles in spherical coordinates.')
plt.show()