import os

dirs = os.listdir('../input/')

for d in dirs:

  print(d)
print('Reading stage1_labels.csv...')

import csv

import numpy as np

with open('../input/stage1_labels.csv', 'r') as f:

  reader = csv.reader(f, delimiter=',', quotechar='\'')

  stage1_labels = list(reader)



print('Creating sample_labels csv...')

sample_labels = ('id', 'cancer')

dirs = os.listdir('../input/sample_images/')

for d in dirs:

  for s1_label in stage1_labels:

    if(s1_label[0] == d):

      sample_labels = np.append(sample_labels, s1_label)



sample_labels = np.reshape(sample_labels, (-1,2))

print(sample_labels)

print('done.')