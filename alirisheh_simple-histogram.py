import numpy as np

import os

import pandas as pd

import re

import sys

from scipy.io import loadmat



def load_challenge_data(filename):



    x = loadmat(filename)

    data = np.asarray(x['val'], dtype=np.float64)



    new_file = filename.replace('.mat', '.hea')

    input_header_file = os.path.join(new_file)



    with open(input_header_file, 'r') as f:

        header_data = f.readlines()



    return data, header_data





def save_challenge_predictions(output_directory, filename, scores, labels, classes):



    recording = os.path.splitext(filename)[0]

    new_file = filename.replace('.mat', '.csv')

    output_file = os.path.join(output_directory, new_file)



    labels=np.asarray(labels,dtype=np.int)

    scores=np.asarray(scores,dtype=np.float64)



    # Include the filename as the recording number

    recording_string = '#{}'.format(recording)

    class_string = ','.join(classes)

    label_string = ','.join(str(i) for i in labels)

    score_string = ','.join(str(i) for i in scores)



    with open(output_file, 'w') as f:

        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')





# Find unique number of classes

def get_classes(input_directory, files):



    classes = set()

    for f in files:

        g = f.replace('.mat', '.hea')

        input_file = os.path.join(input_directory, g)

        with open(input_file, 'r') as f:

            for lines in f:

                if lines.startswith('#Dx'):

                    tmp = lines.split(': ')[1].split(',')

                    for c in tmp:

                        classes.add(c.strip())



    return sorted(classes)

input_directory = '../input/12-lead-ecg/Training_WFDB/'

output_directory = ''



# Find files.

input_files = []

for f in os.listdir(input_directory):

    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):

        input_files.append(f)



classes = get_classes(input_directory, input_files)





# Iterate over files.

print('Extracting 12ECG features...')

num_files = len(input_files)

X = []

Y = []

for i, f in enumerate(input_files):

    tmp_input_file = os.path.join(input_directory, f)





    data, header_data = load_challenge_data(tmp_input_file)



    X.append(np.array(data))

    Y.append([it.split(': ')[1].rstrip() for it in header_data if re.search('Dx:.*', it)][0])



print('Done.')
import matplotlib.pyplot as plt

total_mean = {}

for col in np.unique(Y):

    total_mean.update({str(col): []})

for r in np.unique(Y):

    row_ix = np.where(np.array(Y) == r)

#     print(np.mean(X[1][2]))

    for index in row_ix[0]:

        total_mean[str(r)].append(np.mean(np.array(X[index]), axis=1))
plt.hist(total_mean['164884008'])

plt.show()
plt.hist(total_mean['164889003'])

plt.show()
plt.hist(total_mean['429622005'])

plt.show()
plt.hist(total_mean['164889003,59118001'])

plt.show()