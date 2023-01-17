import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
import numpy as np
df = pd.read_csv('../input/air-bnb-ny-2019/AB_NYC_2019.csv',)
df.info()
df.head()
mygroup = df[["neighbourhood","price"]].groupby("neighbourhood").mean().reset_index()
print("A number of dateTime : ",len(mygroup))
mygroup
df = mygroup.set_index('neighbourhood')
df
Z = hierarchy.linkage(df, 'ward')
fig = plt.figure(figsize=(60, 50))
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=df.index)
plt.show()