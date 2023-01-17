import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import pandas as pd

import numpy as np

import math

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 

warnings.filterwarnings('ignore')
# load metadata

t1 = time.time()

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

t2 = time.time()

print('Elapsed time:', t2-t1)
df.head()
df.isnull().sum()
df.info()
df = df[['authors','title','journal']]

df.head()
#journal top 30

df.journal.value_counts()[0:30].plot(kind='bar')

plt.grid()

plt.show()
#title top 30

df.title.value_counts()[0:30].plot(kind='bar')

plt.show()
"""import tarfile



tar = tarfile.open("/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv.tar")

tar.extractall()

tar.close()"""
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/0089aa4b17549b9774f13a9e2e12a84fc827d60b.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/0049ba8861864506e1e8559e7815f4de8b03dbed.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/0022796bb2112abd2e6423ba2d57751db06049fb.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/02522a1b4512e4d2880a6a551a8dff589d4cb393.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/014e31dce7e3f2b1a7020a5debfbf228182f8b5e.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/0036b28fddf7e93da0970303672934ea2f9944e7.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
print(json_file['metadata']['title'])

print('\nAbstract: \n\n', json_file['abstract'])