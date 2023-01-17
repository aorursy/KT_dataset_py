# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from subprocess import check_output
print(check_output(["ls","../input/gutenberg-dammit"]).decode("utf8"))
from subprocess import check_output
print(check_output(["ls","../input/gutenberg-dammit/gutenberg-dammit-files"]).decode("utf8"))
import json
from pandas.io.json import json_normalize
metadata = pd.read_json("../input/gutenberg-dammit/gutenberg-dammit-files/gutenberg-metadata.json")
print(metadata.shape)
metadata.head()
metadata.head(50)
f = open("../input/gutenberg-dammit/gutenberg-dammit-files/000/00014.txt", "r")
print(f.read())
metadata.info()
metadata['Subject'].value_counts()
n = 20
metadata['Subject'].value_counts()[:n].index.tolist()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
metadata['Subject'].value_counts()[:n].plot(ax=ax, kind='bar')
metadata['Subject'].value_counts()[n:].index.tolist()
n = 20
metadata['Author'].value_counts()[:n].index.tolist()
n=20

fig, ax = plt.subplots()
metadata['Author'].value_counts()[:n].plot(ax=ax, kind='bar')
metadata.tail()
f = open("../input/gutenberg-dammit/gutenberg-dammit-files/000/00013.txt", "r")
print(f.read())
lewis_df = pd.read_fwf(r'../input/gutenberg-dammit/gutenberg-dammit-files/000/00013.txt', header=None, names=['Verse'])
lewis_df.head()
lewis_df.head(50)
from __future__ import unicode_literals
import spacy
nlp = spacy.load('en')

#used the first verse of Lewis's book as data
doc = nlp("Just the place for a Snark! the Bellman cried,As he landed his crew with care; Supporting each man on the top of the tide By a finger entwined in his hair.")