#Ignore the seaborn warnings.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
#Import data and see what states it has values for so far.
df = pd.read_csv('../input/primary_results.csv')
df.state.unique()
#Create a new dataframe that holds votes by state and the fraction of total votes(democrat + republican) that a candidate recieved and pare them down to only those that are still in the race as of 2 March.

votesByState = [[candidate, state] for candidate in df.candidate.unique() for state in df.state.unique()]
