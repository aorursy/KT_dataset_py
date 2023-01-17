import sys

import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import scipy as sp

import minepy as mp

from sklearn import cross_validation, ensemble, preprocessing

from statsmodels.graphics import mosaicplot
# We import the documents.

df_Data_Train = pd.read_csv("C:/Users/nico/Documents/Cours/Machine-Learning Amir Sani/train.csv", sep=";") 

df_Data_Test = pd.read_csv("C:/Users/nico/Documents/Cours/Machine-Learning Amir Sani/test.csv", sep=";") 