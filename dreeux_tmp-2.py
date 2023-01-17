import warnings
warnings.filterwarnings("ignore");

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats

df = pd.read_csv("D:/LOCAL/KAGGLE/US_ELECTION/2016_presidential_election/primary_results.csv")

df.head(); df.tail(3)

df.index

df.columns

df.values

df.describe

df.state.unique()

votesByState = [
    
                [candidate, state] 

                for candidate in df.candidate.unique()
                
                for state in df.state.unique()
                
                ]
print(votesByState)