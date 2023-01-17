!pip install https://github.com/LAL/trackml-library/archive/master.zip
import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

from trackml.dataset import load_dataset
def dbcluster(eps):

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

    cl = DBSCAN(eps=eps, min_samples=1, algorithm='kd_tree')

    labels = cl.fit_predict(X)   

    return labels
path_to_test = "../input/trackml-particle-identification/test"

test_dataset_submissions = []



for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

    labels = dbcluster(0.00715)

    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))

    one_submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)

    test_dataset_submissions.append(one_submission)

    print(event_id)



submission = pd.concat(test_dataset_submissions, axis=0)

submission.to_csv('submission.csv.gz', index=False, compression='gzip')