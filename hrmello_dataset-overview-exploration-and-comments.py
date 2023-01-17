import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
#read csv
training = pd.read_csv('../input/training_set.csv')
training.sample(5)
meta_training = pd.read_csv("../input/training_set_metadata.csv")
meta_training.sample(5)
unique_targets = meta_training.target.unique()
print ("There are {} unique targets.".format(len(unique_targets)))
print (unique_targets)
objects_per_target = pd.DataFrame(meta_training.groupby("target", as_index = False)["object_id"].count())
objects_per_target = objects_per_target.rename(columns = {"object_id": "num_of_objects"})
fig = plt.figure(figsize=(10,8))
sb.barplot(x =objects_per_target.target, y = objects_per_target.num_of_objects);
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for class_target in unique_targets:
    class_used = meta_training[meta_training.target == class_target]
    ax.scatter(x = class_used.gal_l, y = class_used.gal_b, alpha = 1)
plt.xlabel("Galactical Longitude(°)", fontsize = 15)
plt.ylabel("Galactical Latitude(°)", fontsize = 15);
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for class_target in unique_targets:
    class_used = meta_training[meta_training.target == class_target]
    ax.scatter(class_used.gal_l,class_used.gal_b, alpha = .1)
ax.set_xlabel("Galactical Longitude (°)", fontsize = 15)
ax.set_ylabel("Galactical Latitude (°)", fontsize = 15);
condition = (round(meta_training["gal_b"]).isin(range(38,48)) & round(meta_training["gal_l"]).isin(range(226,238)))\
            |(round(meta_training["gal_b"]).isin(range(-56,-52)) & round(meta_training["gal_l"]).isin(range(220,226)))\
            |(round(meta_training["gal_b"]).isin(range(-66,-56)) & round(meta_training["gal_l"]).isin(range(165,178)))\
            |(round(meta_training["gal_b"]).isin(range(-55,-45)) & round(meta_training["gal_l"]).isin(range(315,323)))\
            |(round(meta_training["gal_b"]).isin(range(-77,-65)) & round(meta_training["gal_l"]).isin(range(322,332)))
five_point = meta_training.loc[condition]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
for class_target in unique_targets:
    class_used = five_point[five_point.target == class_target]
    ax.scatter(class_used.gal_l,class_used.gal_b, alpha = .1)
ax.set_xlabel("Galactical Longitude (°)", fontsize = 15)
ax.set_ylabel("Galactical Latitude (°)", fontsize = 15);
objects_per_target_five = pd.DataFrame(five_point.groupby("target", as_index = False)["object_id"].count())
objects_per_target_five = objects_per_target_five.rename(columns = {"object_id": "num_of_objects"})
fig = plt.figure(figsize=(10,8))
sb.barplot(x =objects_per_target_five.target, y = objects_per_target_five.num_of_objects);
plt.xlabel("Class", fontsize = 15)
plt.ylabel("Number of sources", fontsize = 15);
ddf_counts = pd.DataFrame(meta_training.groupby("ddf", as_index = False)["object_id"].count())
ddf_counts
ddf_counts_five = pd.DataFrame(five_point.groupby("ddf", as_index = False)["object_id"].count())
ddf_counts_five
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(meta_training.gal_l,meta_training.gal_b, c = meta_training.distmod, s = 7, cmap = 'Reds');
ax.set_xlabel("Galactical Longitude (°)", fontsize = 15)
ax.set_ylabel("Galactical Latitude (°)", fontsize = 15);
fig = plt.figure(figsize=(10,8))
sb.distplot(meta_training[~np.isnan(meta_training.distmod)].distmod);
# plt.xlabel("Distance to the source")
plt.ylabel("Frequency", fontsize = 15);
plt.xlabel("Distmod", fontsize = 15);
meta_training.distmod.corr(meta_training.hostgal_photoz)
fig = plt.figure(figsize=(10,8))
plt.scatter(meta_training.distmod, meta_training.hostgal_photoz, s = 1);
plt.xlabel("Distmod", fontsize = 15);
plt.ylabel("Host Galaxy Photometric Redshift", fontsize = 15);
fig = plt.figure(figsize=(10,8))
plt.scatter(meta_training.distmod, meta_training.hostgal_photoz_err, s = 1);
plt.xlabel("Distmod", fontsize = 15);
plt.ylabel("Host Galaxy Photometric Redshift Error", fontsize = 15);
fig = plt.figure(figsize=(10,8))
plt.scatter(meta_training.distmod, meta_training.hostgal_specz, s = 1);
plt.xlabel("Distmod", fontsize = 15);
plt.ylabel("Host Galaxy Spectroscopic Redshift", fontsize = 15);
#first we should split the dataset between galactic and extragalactic sources
galactic = meta_training[meta_training.hostgal_photoz == 0]
galactic.sample(5)
extragalactic = meta_training[meta_training.hostgal_photoz != 0]
extragalactic.sample(5)
#for convinience while making the plots, I'll create a variable to reference the galactic/extragalactic feature. 1 for extragalactic and 0 for galactic
meta_training['extra'] =  0
meta_training.loc[meta_training.hostgal_photoz != 0, 'extra'] = 1

# plt.hist(meta_training[meta_training.extra == 0].target, label = 'Galactic')

meta_training['target'] = meta_training['target'].astype('category',copy=False)

grid = sb.FacetGrid(data = meta_training, hue = 'extra', height =7)
grid.map(sb.countplot, 'target')
for ax in grid.axes.ravel():
    ax.legend()
plt.xlabel("Class", fontsize = 15)
plt.ylabel("Counts", fontsize = 15);
fig = plt.figure(figsize=(10,8))
plt.scatter(x = galactic.gal_l, y = galactic.gal_b, alpha = 0.5, label = 'galactic')
plt.scatter(x = extragalactic.gal_l, y = extragalactic.gal_b, alpha = 0.1, label = 'extragalactic');
plt.xlabel("Galactic longitude (°)", fontsize = 15)
plt.ylabel("Galactic latitude (º)", fontsize = 15)
plt.legend();
merged = training.merge(meta_training, on = "object_id")
merged.sample(5)
grid = sb.FacetGrid(data = merged, height =7)
grid.map(sb.countplot, 'passband')
for ax in grid.axes.ravel():
    ax.legend()
plt.xlabel("Passband", fontsize = 15)
plt.ylabel("Counts", fontsize = 15);
unique_sources = merged.object_id.unique()
obj_data = merged[merged.object_id == unique_sources[0]]
obj_data.head(5)
unique_passbands = merged.passband.unique()

fig = plt.figure(figsize=(10,8))
for passband in unique_passbands:
    specific_pb = obj_data[obj_data.passband == passband]
    plt.scatter(specific_pb.mjd, specific_pb.flux, label = passband)
plt.xlabel("MJD (in days from November 17, 1858)", fontsize = 15)
plt.ylabel("Flux", fontsize = 15)
plt.legend();
window_objdata = obj_data[(obj_data.mjd > 60100) & (obj_data.mjd<60300)]
fig = plt.figure(figsize=(15,8))
for passband in unique_passbands:
    specific_pb = window_objdata[window_objdata.passband == passband]
    plt.plot(specific_pb.mjd, specific_pb.flux, label = passband)
plt.xlabel("MJD (in days from November 17, 1858)", fontsize = 15)
plt.ylabel("Flux", fontsize = 15)
plt.legend();
#get objects from class 92
class_92_objs = merged[merged.target == 92]
#unique objects from class 92
unique_sources_92 = class_92_objs.object_id.unique()
#get data from one specific object from class 92 (elemment 0 is the same from before)
obj_data = merged[merged.object_id == unique_sources_92[4]]
#get one time window to observe and plot
window_objdata = obj_data[(obj_data.mjd > 60100) & (obj_data.mjd<60300)]
fig = plt.figure(figsize=(15,8))
for passband in unique_passbands:
    specific_pb = window_objdata[window_objdata.passband == passband]
    plt.plot(specific_pb.mjd, specific_pb.flux, label = passband)
plt.xlabel("MJD (in days from November 17, 1858)", fontsize = 15)
plt.ylabel("Flux", fontsize = 15)
plt.legend();
#get objects from class 52
class_52_objs = merged[merged.target == 52]
#unique objects from class 52
unique_sources_52 = class_52_objs.object_id.unique()
#get data from one specific object from class 52
obj_data = merged[merged.object_id == unique_sources_52[0]]
#get one time window to observe and plot
window_objdata = obj_data[(obj_data.mjd > 60100) & (obj_data.mjd<60300)]
fig = plt.figure(figsize=(15,8))
for passband in unique_passbands:
    specific_pb = window_objdata[window_objdata.passband == passband]
    plt.plot(specific_pb.mjd, specific_pb.flux, label = passband)
plt.xlabel("MJD (in days from November 17, 1858)", fontsize = 15)
plt.ylabel("Flux", fontsize = 15)
plt.legend();
#get data from one specific object from class 52
obj_data = merged[merged.object_id == unique_sources_52[2]]
#get one time window to observe and plot
window_objdata = obj_data[(obj_data.mjd > 60100) & (obj_data.mjd<60300)]
fig = plt.figure(figsize=(15,8))
for passband in unique_passbands:
    specific_pb = window_objdata[window_objdata.passband == passband]
    plt.plot(specific_pb.mjd, specific_pb.flux, label = passband)
plt.xlabel("MJD (in days from November 17, 1858)", fontsize = 15)
plt.ylabel("Flux", fontsize = 15)
plt.legend();