import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
%matplotlib inline
df = pd.read_csv("../input/training_set_metadata.csv")
df.head()
ddf = df[df['ddf']==1] # Deep Drilling Fields
wfd = df[df['ddf']==0] # Wide-Fast-Deep survey
fig, axes = plt.subplots(2, 1, figsize=(10, 12),subplot_kw={'projection': 'aitoff'})
axes[0].scatter(x=(ddf['ra']-180)*math.pi/180, y=ddf['decl']*math.pi/180, c=ddf['hostgal_specz'], cmap='plasma', s=2)
axes[0].set_title("DDF survey area")
axes[1].scatter(x=(wfd['ra']-180)*math.pi/180, y=wfd['decl']*math.pi/180, c=wfd['hostgal_specz'], cmap='plasma', s=2)
axes[1].set_title("WFD survey area")
ingal = df[df['hostgal_specz']==0] # inside our galaxy
outgal = df[df['hostgal_specz']>0] # outside our galaxy
fig, axes = plt.subplots(2, 1, figsize=(10, 12),subplot_kw={'projection': 'aitoff'})
axes[0].scatter(x=(ingal['ra']-180)*math.pi/180, y=ingal['decl']*math.pi/180, c=ingal['hostgal_specz'], cmap='plasma', s=2)
axes[0].set_title("Inside our galaxy")
axes[1].scatter(x=(outgal['ra']-180)*math.pi/180, y=outgal['decl']*math.pi/180, c=outgal['hostgal_specz'], cmap='plasma', s=2)
axes[1].set_title("Outside our galaxy")
print(ingal['target'].sort_values().unique())
print(outgal['target'].sort_values().unique())
ingal_targets = pd.DataFrame(ingal.groupby('target')['object_id'].nunique())
ingal_targets.reset_index(level=0, inplace=True)
ingal_targets.columns = ['target', 'count']
ingal_targets = ingal_targets.assign(our_galaxy = 1)
outgal_targets = pd.DataFrame(outgal.groupby('target')['object_id'].nunique())
outgal_targets.reset_index(level=0, inplace=True)
outgal_targets.columns = ['target', 'count']
outgal_targets = outgal_targets.assign(our_galaxy = 0)
targets = pd.concat([ingal_targets,outgal_targets], ignore_index=True)
targets = targets.sort_values(by=['target'])
targets = targets.reset_index(drop=True)
targets
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.bar(targets[targets['our_galaxy']==1].index, targets[targets['our_galaxy']==1]['count'], label="Inside our galaxy")
ax.bar(targets[targets['our_galaxy']==0].index, targets[targets['our_galaxy']==0]['count'], label="Outside our galaxy")
ax.legend(loc='upper left')
ax.set_title("Target types inside/outside of our galaxy")
ax.set_xticks(targets.index)
ax.set_xticklabels(targets['target']);
ax.set_xlabel("Target")
ax.set_ylabel("Numer of occurrence in the dataset")
outgal = outgal.assign(meas_diff = abs(outgal['hostgal_specz']-outgal['hostgal_photoz']))
fig, axes = plt.subplots(2, 1, figsize=(10, 12),subplot_kw={'projection': 'aitoff'})
axes[0].scatter(x=(outgal['ra']-180)*math.pi/180, y=outgal['decl']*math.pi/180, c=outgal['meas_diff'], cmap='Blues', s=2)
axes[0].set_title("Where are the biggest differences")
axes[1].scatter(x=(outgal['ra']-180)*math.pi/180, y=outgal['decl']*math.pi/180, c=outgal['hostgal_photoz_err'], cmap='Blues', s=2)
axes[1].set_title("Where are the biggest errors")
sns.pairplot(outgal[['hostgal_specz','hostgal_photoz','hostgal_photoz_err','mwebv','target','meas_diff']], hue='target')
