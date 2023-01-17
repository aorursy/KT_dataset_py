import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
%matplotlib inline

# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)
# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users.csv')
test_users = pd.read_csv('../input/test_users.csv')
