%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 
               3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 
               6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
DATA_DIR = os.path.join('..', 'input')
train_path = os.path.join(DATA_DIR, 'bildstein_station1_xyz_intensity_rgb.h5')
test_path = os.path.join(DATA_DIR, 'domfountain_station1_xyz_intensity_rgb.h5')
def read_as_df(in_path, only_valid = True):
    with h5py.File(in_path) as h:
        cur_df = pd.DataFrame({k: h[k].value for k in h.keys()})
        if only_valid:
            return cur_df[cur_df['class']>0]
        else:
            return cur_df
train_df = read_as_df(train_path)
test_df = read_as_df(test_path)
print('Train Points', train_df.shape, 'Testing Points', test_df.shape)
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
in_vars = ['intensity', 'r', 'g', 'b']
def fit_and_predict(in_class, bonus_vars = None):
    c_vars = in_vars
    if bonus_vars is not None:
        c_vars += bonus_vars
    in_class.fit(train_df[c_vars], train_df['class'])
    c_pred = in_class.predict(test_df[c_vars])
    labels = [label_names.get(k) for k in range(test_df['class'].max()+1)]
    cm = confusion_matrix(test_df['class'].values, c_pred, labels = range(test_df['class'].max()+1))
    fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
    sns.heatmap(cm,
                     ax = ax1,
                     cbar = True, 
                     fmt = 'd', 
                     annot = True,
                     xticklabels=labels, 
                     yticklabels=labels)
    try:
        model_name = c_class.__class__.__name__
    except:
        model_name = repr(c_class)
    out_str = 'Model: {}\n Accuracy {:2.2f}%'.format(model_name, 
                                                        100*accuracy_score(test_df['class'].values, c_pred))
    print(out_str)
    ax1.set_title(out_str)
    return c_pred, in_class
from sklearn.dummy import DummyClassifier
c_class = DummyClassifier()
fit_and_predict(c_class);
from sklearn.neighbors import KNeighborsClassifier
c_class = KNeighborsClassifier(2, n_jobs = 4)
fit_and_predict(c_class);
from sklearn.linear_model import LogisticRegression
c_class = LogisticRegression(n_jobs = 4)
fit_and_predict(c_class);
from sklearn.ensemble import RandomForestClassifier
c_class = RandomForestClassifier(n_jobs = 4)
fit_and_predict(c_class);
from sklearn.ensemble import ExtraTreesClassifier
c_class = ExtraTreesClassifier(n_jobs = 4)
fit_and_predict(c_class);
from sklearn.ensemble import RandomForestClassifier
c_class = RandomForestClassifier(n_jobs = 4)
fit_and_predict(c_class, bonus_vars = ['x', 'y', 'z']);
