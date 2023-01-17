import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

%matplotlib inline

p = sns.color_palette()



os.listdir('../input')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
for d in os.listdir('../input/sample_images'):

    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))

print('----')

print('Total patients {} Total DCM files {}'.format(len(os.listdir('../input/sample_images')), 

                                                      len(glob.glob('../input/sample_images/*/*.dcm'))))
ppatient_sizes = [len(os.listdir('../input/sample_images/'+ d)) for d in os.listdir('../input/sample_images/')]



plt.hist(patient_sizes, color=p[2])

plt.xlabel('Number Of DCIM files')

plt.ylabel('Number of patients')

plt.title('Histogram of DCIM Files per patient count')