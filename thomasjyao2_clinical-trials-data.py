import os 

import seaborn as sns

os.chdir("../input/clinical-data/")



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")

from textpack import tp

import re

from collections import Counter

from matplotlib import rc

import squarify



# Read in Clinical trials data

df=pd.read_csv("Search_2.csv")

df.head(n = 5)
!pip install textpack

os. getcwd() 
#import IPython

!jupyter nbconvert clinical-trials-data.ipynb --to=html --output-dir='/kaggle/output/working' --output = Output2.html

#! jupyter nbconvert my_notebook.ipynb --to=html --output=Output.html



os.chdir("/kaggle/input/clinical")


