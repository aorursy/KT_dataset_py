import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

import matplotlib.animation as animation

from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')



try:

    t_file = pd.read_csv('../input/questions.csv', encoding='ISO-8859-1')

    print('File load: Success')

except:

    print('File load: Failed')
t_file = t_file.dropna()

print(t_file.head())
from nltk.corpus import stopwords

stop = stopwords.words('english')

print(stop)
t_file['question1'] = t_file['question1'].str.lower().str.split()

t_file['question2'] = t_file['question2'].str.lower().str.split()

t_file['question1'] = t_file['question1'].apply(lambda x: [item for item in x if item not in stop])

t_file['question2'] = t_file['question2'].apply(lambda x: [item for item in x if item not in stop])
common_words =list(set(t_file['question1'][1]).intersection(t_file['question2'][1]))



print(common)