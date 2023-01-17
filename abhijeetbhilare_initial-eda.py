from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd

import seaborn as sns
import missingno as msno
from scipy import stats
from pandas_profiling import ProfileReport
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/advertisement-marketing/Advertising.csv", index_col=0)
df.head()
ProfileReport(df)
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
sns.pairplot(df)
df1 = df.loc[df['sales'] == df["sales"].max()]
df1.head()
df1 = df.loc[df['sales'] == df["sales"].min()]
df1.head()