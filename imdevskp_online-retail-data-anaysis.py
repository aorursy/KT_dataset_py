import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
df = pd.read_excel('../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')

df.head()
df.info()
df.describe(include='all')