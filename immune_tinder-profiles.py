# 🎨 Justin Rayfield Faler

# 🧬 Description: Tinder

# 🧪 https://github.com/Jfaler



import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import gensim

import scipy

import numpy

import json

import nltk

import sys

import csv

import os



print(os.listdir("../input"))



# Read xlsx file & fix bad lines

tinder_profile_df = pd.read_excel('../input/tdf-v2/tinder_data_final.xlsx', error_bad_lines=False, delimiter='\t')



# Print dataframe info

tinder_profile_df.info()



# This will print the number of rows and columns that will be used to train

print("Shape of train set : ",tinder_profile_df.shape)



tinder_profile_df.head()