import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
data=pd.read_csv('../input/birandom1107.csv',header=None)
values=data.values
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(values)
labels = gmm.predict(values)
labels
