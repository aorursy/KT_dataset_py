import os



import numpy as np

import pandas as pd



from trackml.dataset import load_event

from trackml.randomize import shuffle_hits

from trackml.score import score_event



import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

%matplotlib inline
event_prefix = 'event000001000'

hits, cells, particles, truth = load_event(os.path.join('../input/train_1', event_prefix))



mem_bytes = (hits.memory_usage(index=True).sum() 

             + cells.memory_usage(index=True).sum() 

             + particles.memory_usage(index=True).sum() 

             + truth.memory_usage(index=True).sum())

print('{} memory usage {:.2f} MB'.format(event_prefix, mem_bytes / 2**20))
hits.head()
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
fig = plt.figure(figsize=(12, 12))

ax = fig.add_subplot(111, projection='3d')

for volume in volumes:

    v = hits[hits.volume_id == volume]

    ax.scatter(v.z, v.x, v.y, s=1, label='volume {}'.format(volume), alpha=0.5)

ax.set_title('Hit Locations')

ax.set_xlabel('Z (millimeters)')

ax.set_ylabel('X (millimeters)')

ax.set_zlabel('Y (millimeters)')

plt.show()
hits_sample = hits.sample(8000)

sns.pairplot(hits_sample, hue='volume_id', size=8)

plt.show()
particles.head()
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
g = sns.jointplot(particles.vx, particles.vy,  s=3, size=12)

g.ax_joint.cla()

plt.sca(g.ax_joint)



n_hits = particles.nhits.unique()

for n_hit in n_hits:

    p = particles[particles.nhits == n_hit]

    plt.scatter(p.vx, p.vy, s=3, label='Hits {}'.format(n_hit))



plt.xlabel('X (mm)')

plt.ylabel('Y (mm)')

plt.legend()

plt.show()
g = sns.jointplot(particles.vz, particles.vy,  s=3, size=12)

g.ax_joint.cla()

plt.sca(g.ax_joint)



n_hits = particles.nhits.unique()

for n_hit in n_hits:

    p = particles[particles.nhits == n_hit]

    plt.scatter(p.vz, p.vy, s=3, label='Hits {}'.format(n_hit))



plt.xlabel('Z (mm)')

plt.ylabel('Y (mm)')

plt.legend()

plt.show()
fig = plt.figure(figsize=(12, 12))

ax = fig.add_subplot(111, projection='3d')

for charge in [-1, 1]:

    q = particles[particles.q == charge]

    ax.scatter(q.vz, q.vx, q.vy, s=1, label='Charge {}'.format(charge), alpha=0.5)

ax.set_title('Sample of 1000 Particle initial location')

ax.set_xlabel('Z (millimeters)')

ax.set_ylabel('X (millimeters)')

ax.set_zlabel('Y (millimeters)')

ax.legend()

plt.show()
p_sample = particles.sample(8000)

sns.pairplot(p_sample, vars=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'nhits'], hue='nhits', size=8)

plt.show()
# Get particle id with max number of hits in this event

particle = particles.loc[particles.nhits == particles.nhits.max()].iloc[0]

particle2 = particles.loc[particles.nhits == particles.nhits.max()].iloc[1]



# Get points where the same particle intersected subsequent layers of the observation material

p_traj_surface = truth[truth.particle_id == particle.particle_id][['tx', 'ty', 'tz']]

p_traj_surface2 = truth[truth.particle_id == particle2.particle_id][['tx', 'ty', 'tz']]



p_traj = (p_traj_surface

          .append({'tx': particle.vx, 'ty': particle.vy, 'tz': particle.vz}, ignore_index=True)

          .sort_values(by='tz'))

p_traj2 = (p_traj_surface2

          .append({'tx': particle2.vx, 'ty': particle2.vy, 'tz': particle2.vz}, ignore_index=True)

          .sort_values(by='tz'))



fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



ax.plot(

    xs=p_traj.tx,

    ys=p_traj.ty,

    zs=p_traj.tz, marker='o')

ax.plot(

    xs=p_traj2.tx,

    ys=p_traj2.ty,

    zs=p_traj2.tz, marker='o')



ax.set_xlabel('X (mm)')

ax.set_ylabel('Y (mm)')

ax.set_zlabel('Z  (mm) -- Detection layers')

plt.title('Trajectories of two particles as they cross the detection surface ($Z$ axis).')

plt.show()
# Imports

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from trackml.dataset import load_event, load_dataset

from trackml.score import score_event

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier



# Change the directories as per the location on your PC.

path_to_train = "../input/train_1"

path_to_test = "../input/test"

# This event is in Train_1

event_prefix = "event000001000"



# Loading Event

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))



######################################All the main functions and classes are defined below###########################################################

class Clusterer(object):



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



        standard_scaler = StandardScaler()

        X = standard_scaler.fit_transform(hits[['x2', 'y2', 'z2']].values)

        

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



def get_train_sample(path_to_data, event_names):

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



#The main class in which all functions are implemented implementing KNeighboursClassifier



def one_event_submission(event_id, hits, labels):

    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))

    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)

    return submission



######################################All the implementations are below###########################################################





start_event_id = 1000

n_train_samples = 5

train_event_names = ["event0000{:05d}".format(i) for i in range(start_event_id, start_event_id+n_train_samples)]

train_data = get_train_sample(path_to_train, train_event_names)



# Fit the training data in our try_model



try_model = Clusterer()

try_model.fit(train_data)





event_path = os.path.join(path_to_train, "event0000{:05d}".format(start_event_id + n_train_samples + 1))

hits, cells, particles, truth = load_event(event_path)

labels = try_model.predict(hits)

submission = one_event_submission(0, hits, labels)

score = score_event(truth, submission)





load_dataset(path_to_train, skip=1000, nevents=5)





dataset_submit = []

dataset_scores = []





for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=1000, nevents=5):



    labels = try_model.predict(hits)

    # Prepare submission for an event

    one_submission = one_event_submission(event_id, hits, labels)

    dataset_submit.append(one_submission)

    # Score for the event

    score = score_event(truth, one_submission)

    dataset_scores.append(score)

    print("Score for event %d: %.3f" % (event_id, score))

    

print('Mean score: %.3f' % (np.mean(dataset_scores)))

test_dataset_submit = []

create_submission = True # False for not generating a submission file 





if create_submission:

    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):



        # Track pattern recognition

        labels = try_model.predict(hits)

        # Prepare submission for an event

        one_submission = one_event_submission(event_id, hits, labels)

        test_dataset_submit.append(one_submission)

        print('Event ID: ', event_id)



    # Create submission file

    submission = pd.concat(test_dataset_submit, axis=0)

    submission.to_csv('submission.csv.gz', index=False, compression='gzip')