%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
# Change this according to your directory preferred setting
path_to_train = "../input/train_1"
# This event is in Train_1
event_prefix = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))
hits.head()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

"""
updated - added self.rz_scale
"""
class Clusterer(object):
    
    def __init__(self, eps):
        self.eps = eps
        self.rz_scale = 1
        
    
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
        X[:,2] = X[:,2] * self.rz_scale
        
        return X
    
    
    def predict(self, hits, rz_scale=1):
        
        self.rz_scale = rz_scale
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=1, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
        return labels
"""
updated - added to predict: rz_scale=1.5
"""
model = Clusterer(eps=0.008)
labels = model.predict(hits, rz_scale=1.5)
print(labels)
def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission
submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)
print("Your score: ", score)
dataset_submissions = []
dataset_scores = []

for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
        
    # Track pattern recognition
    model = Clusterer(eps=0.008)
    labels = model.predict(hits, rz_scale=1.5)
        
    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)
    
    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)
    
    print("Score for event %d: %.3f" % (event_id, score))
    
print('Mean score: %.3f' % (np.mean(dataset_scores)))
path_to_test = "../input/test"
test_dataset_submissions = []

create_submission = True # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        model = Clusterer(eps=0.008)
        labels = model.predict(hits, rz_scale=1.5)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submussion = pd.concat(test_dataset_submissions, axis=0)
    submussion.to_csv('submission.csv', index=False)
