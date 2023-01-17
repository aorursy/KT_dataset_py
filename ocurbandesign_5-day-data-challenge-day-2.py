import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/train_parties.csv")
data.info()
data['num_complaints'].head()
data['num_complaints'].describe()
sns.distplot(data['num_complaints'],kde=False, bins =2).set_title('Number of Noise Complaints Histogram')