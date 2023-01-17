import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')
print(train.shape)
print(train_meta.shape)
def view_target(n):
    obj_id = np.random.choice(train_meta.object_id[train_meta.target == n].values)
    obj_df = train[train.object_id == obj_id]
    fig, axes = plt.subplots(6,1, figsize=(10, 10))
    axes[0].set_title(f'Class_{n}')
    for i, ax in enumerate(axes):
        ax.scatter(obj_df.mjd[obj_df.passband == i].values, obj_df.flux[obj_df.passband == i].values, alpha=0.5)    
        ax.scatter(obj_df.mjd[obj_df.passband == i].values, obj_df.flux_err[obj_df.passband == i].values, alpha=0.5)
targets = sorted(train_meta.target.unique())
print(f"Targets: {targets}")
for t in targets:
    view_target(t)