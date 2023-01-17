import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import json
%matplotlib inline 
# disable scientific notat
np.set_printoptions(suppress=True)
#Utility method to read the data pickle
def read_dataset(filename):
    return pd.read_pickle(filename)
def create_label_pickle(JSONFILENAME, OUTPUTFILENAME):
    with open(JSONFILENAME) as data_file:
        data = json.load(data_file)
    dataset = []
    annotations = data['annotations']
    for annotation in annotations:
        labels = list(annotation['labelId'])
        for label in labels:
            a = {}
            a['imageId'] = annotation['imageId']
            a['labelId'] = label
            dataset.append(a)
    panda = pd.DataFrame(dataset)
    print(panda.head())
    panda.to_pickle(OUTPUTFILENAME)
TRAIN_FILE = '../input/train.json'
TRAIN_LABEL_PICKLE = './train_label.pickle'

if os.path.isfile(TRAIN_LABEL_PICKLE) == False:
    create_label_pickle(TRAIN_FILE, TRAIN_LABEL_PICKLE)
dataset = read_dataset(TRAIN_LABEL_PICKLE)
dataset = pd.DataFrame(dataset, dtype='int32')
number_of_labels = dataset['labelId'].nunique() # Number of distinct labels
maximum_label_id = max(dataset['labelId']) # The maximum labelId value
number_of_images = dataset['imageId'].nunique() # Number of distinct images
print('Number of distinct labels in the dataset : ', number_of_labels)
print('Maximum id if labels in the dataset : ', maximum_label_id)
print('Number of distinct images in the dataset : ', number_of_images)
# Count analysis for images
count_by_image_id = dataset.groupby('imageId')['imageId'].count().reset_index(name="count")
count_by_label_id = dataset.groupby('labelId')['labelId'].count().reset_index(name="count")
# Plot by label counts
print('Images with largest number of labels ')
count_by_image_id.nlargest(5, 'count')
print(' Number of labels versus number of images with that many labels :')
a = count_by_image_id['count'].value_counts().sort_index().plot(kind = 'bar')
# Plot by label counts
print('Labels associated with largest number of images ')
count_by_label_id.nlargest(5, 'count')
count_by_label_id.plot(title='Labels versus how many times they occur')
check_relation = np.zeros((maximum_label_id + 1 ,maximum_label_id + 1)) # adding one because labels are 1 indexed.
# we start by creating a dict with imageId as keys and list of labels as values.
relations = {}
for index, row in dataset.iterrows():
    imageId = row['imageId']
    labelId = row['labelId']
    if imageId in relations:
        # if this imageId is already there, map this label to all other labels already encountered.
        for l in relations[imageId]:
            check_relation[l][labelId] += 1
            check_relation[labelId][l] += 1
    else:
        # add this imageId to dict
        relations[imageId] = []
    # add this label to the imageId label's list
    relations[imageId].append(labelId)
# I am creating a clone here becasue in next few steps I am going to sort the matrix.
# I want to retain the original mappings also.
temp = np.copy(check_relation)
# Revert step in case I screw up later.
#check_relation = np.copy(temp)
check_relation[10, :]
closest_companions = np.argsort(temp, axis=1) # axis = 1 to sort along the rows.
# Here I am filtering just the last three columns which have the highest values. 
# [:,::-1] is to reverse the three values because the values occur in ascending order 
# and I wanted them in descending
closest_companions = (closest_companions[:, -3:])[:,::-1]
print(closest_companions[10])
# the `temp` we created earlier will be used here.
sorted_closest_companions = temp
sorted_closest_companions.sort(axis = 1)
closest_companions_count = (sorted_closest_companions[:, -3:])[:,::-1]
print(closest_companions_count[10])
companion = pd.DataFrame(columns=['companion'],data = closest_companions[1:,0], index = range(1, closest_companions.shape[0]))
companion['labelId'] = range(1, closest_companions.shape[0])
companion.head()
# I am taking closest_companions_count[1:,0] because
# the matrix indexing starts from 0 and we dont have a labelid = 1 in dataset.
# So no use keepin that value
companion_count = pd.DataFrame(dtype='int32',columns=['count'],data = closest_companions_count[1:,0], index = range(1, closest_companions_count.shape[0]))
companion_count['labelId'] = range(1, closest_companions_count.shape[0])
companion_count.head()
ax = companion_count.plot(title='Labels versus how many times they occur', x='labelId', y='count')
count_by_label_id.plot(ax= ax)
merged = pd.merge(companion_count, count_by_label_id, on='labelId')
merged.head()
# renaming columns to something more sensible.
merged.columns = ['companion_count', 'labelId', 'dataset_count']
merged.head()
merged['percentage'] = (merged['companion_count'] / merged['dataset_count'] ) * 100
merged.head()
merged = pd.merge(merged, companion, on='labelId')
merged.head()
merged = merged[['labelId', 'dataset_count', 'companion', 'companion_count', 'percentage']]
merged.head()
# Thanks for reading :) 
