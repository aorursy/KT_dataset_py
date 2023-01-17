import csv
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 999)
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/data.csv')
df.head()
df.describe()
print(df['action_type'].unique())
print(df.combined_shot_type.unique())
