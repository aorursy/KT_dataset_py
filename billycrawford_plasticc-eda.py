import os
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set(style="darkgrid")
train_df = pd.read_csv('../input/training_set.csv')
train_df.head(5)
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_meta.head()
train_meta.describe()
help_dict = {'object_id' : 'Object ID',
                'ra' : 'Right Ascension, corresponding to latitude', 
                'decl' : 'Declination (corresponding to longitude)',
                'gal_l': 'Galactic coordinate 1',
                'gal_b': 'Galactic coordinate 2',
                'ddf': 'Object is observed usning Deep Drilling Field (DDFs have more observations but cover a smaller chunk of sky)',
                'hostgal_specz': 'The spectroscopic redshift of the source8. This is an extremely accurate measure of redshift, provided for the training set and a small fraction of the test set (given as float32 numbers).',
                'hostgal_photoz': 'The photometric redshift of the host galaxy of the astronomical source. While this is meant to be a proxy for hostgal specz, there can be large differences between the two and hostgal photoz should be regarded as a far less accurate version of hostgal specz',
                'hostgal_photoz_err': 'The uncertainty on the hostgal photoz based on LSST survey projections.',
                'distmod': 'The distance (modulus) calculated from the hostgal photoz since this redshift is given for all objects (given as float32 numbers). Computing the distance modulus requires knowledge of General Relativity, and assumed values of the dark energy and dark matter content of the Universe, as mentioned in the introduction section.',
                'mwebv' : 'MW E(B-V): this ‘extinction’ of light is a property of the Milky Way (MW) dust along the line of sight to the astronomical source, and is thus a function of the sky coordinates of the source ra, decl. This is used to determine a passband dependent dimming and reddening of light from astronomical sources as described in subsection 2.1',
                'target': 'The class of the astronomical source. This is provided in the training data.'
               }

plt.figure(figsize = (10,6))
plt.subplot(111,projection ='aitoff')
plt.scatter(x = (train_meta['ra']-180)*math.pi/180, y = (train_meta['decl'])*math.pi/180);
fig, ax = plt.subplots(figsize = (10,6))
sns.countplot(train_meta['ddf']);
fig, ax = plt.subplots(figsize = (10,6))
sns.countplot(train_meta['target']);
df_0 = train_meta.loc[train_meta['ddf']==0]
df_1 = train_meta.loc[train_meta['ddf']==1]


ymax = 1.1* max(list(100*df_1['target'].value_counts()/len(df_1))+list(100*df_0['target'].value_counts()/len(df_0)))

plt.figure(figsize = (20,6))
plt.subplot(121)
ax = sns.barplot(x="target", y="target", data=df_1, estimator=lambda x: len(x) / len(df_1) * 100)
ax.set(ylabel="%", xlabel ='Target')
ax.set_ylim(0,ymax)
plt.title('DDF')
plt.subplot(122)
ax = sns.barplot(x="target", y="target", data=df_0, estimator=lambda x: len(x) / len(df_0) * 100)
ax.set(ylabel="%", xlabel ='Target')
ax.set_ylim(0,ymax)
plt.title('Wide Angle');
df_1['target'].loc[df_1['target'].isin([53,64])].value_counts()
plt.figure(figsize = (10,8))
ax = plt.subplot(111)
ax.scatter(train_meta['hostgal_photoz'],train_meta['hostgal_specz'], marker = '.')
ax.errorbar(train_meta['hostgal_photoz'],train_meta['hostgal_specz'], xerr= None, yerr= train_meta['hostgal_photoz_err'],fmt='none');
model = smf.ols(formula='hostgal_specz ~ hostgal_photoz', data=train_meta)
results = model.fit()
print(results.summary())
plt.figure(figsize = (10,8))
ax = plt.subplot(111)
fig = sm.graphics.plot_fit(results, 1, ax=ax)
train_meta['hostgal_sq_diff'] = (train_meta['hostgal_photoz'] - train_meta['hostgal_specz'])**2
train_meta['hostgal_diff'] = train_meta['hostgal_photoz'] - train_meta['hostgal_specz']
corr = train_meta.corr()

# create a heatmap plot
sns.set(style="white")
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9,as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
model = smf.ols(formula='hostgal_specz ~ hostgal_photoz + hostgal_photoz_err', data=train_meta)
results = model.fit()
print(results.summary())
plt.figure(figsize = (20,8))
ax = plt.subplot(121)
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax = plt.subplot(122)
fig = sm.graphics.plot_fit(results, 2, ax=ax)
model = smf.ols(formula='hostgal_specz ~ hostgal_photoz * hostgal_photoz_err', data=train_meta)
results = model.fit()
print(results.summary())
plt.figure(figsize = (21,8))
ax = plt.subplot(131)
fig = sm.graphics.plot_fit(results, 1, ax=ax)
ax = plt.subplot(132)
fig = sm.graphics.plot_fit(results, 2, ax=ax)
ax = plt.subplot(133)
fig = sm.graphics.plot_fit(results, 3, ax=ax)
cols = list(train_meta)
plt.subplots_adjust(hspace = 2)
plt.figure(figsize = (20,30))
plt.title('Histograms');
for counter, col in enumerate(cols):
    index = counter+1
    ax = plt.subplot(7,2,index)
    ax.hist(train_meta[col].dropna(), bins=10)
    plt.title(col);

train_meta.isna().sum()
train_df.isna().sum()
train_meta['dist_mod_missing'] = train_meta['distmod'].isna()
corr = train_meta.corr()

# create a heatmap plot
sns.set(style="white")
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9,as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
plt.figure(figsize = (10,8))
sns.barplot(train_meta.groupby(['target']).sum().index,train_meta.groupby(['target']).sum()['dist_mod_missing']);
test_df_reader = pd.read_csv('../input/test_set.csv',chunksize = 1000) # use an iterator as otherwise file too large
test_df_reader.get_chunk(5)
# arrays_of_sums = [chunk.isna().sum() for chunk in test_df_reader]
# sum(arrays_of_sums)
test_meta = pd.read_csv('../input/test_set_metadata.csv')
test_meta.head()