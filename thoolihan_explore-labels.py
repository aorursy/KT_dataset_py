!ls -lah ../input/
import pandas as pd
import os

df = pd.read_csv(os.path.join('..', 'input', 'stage_1_train_labels.csv'))
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

df.Target.value_counts().plot(kind='bar', title='Target Frequency')
target_spot = df.Target == 1

total_unique = df.patientId.nunique()
pos = df[target_spot].patientId.nunique() 
boxes = sum(target_spot)
neg = df[~target_spot].patientId.nunique()

print("  {:7,} # positive unique patients".format(pos))
print("+ {:7,} # negative unique patients".format(neg))
print("===========================")
print("  {:7,} # total".format(pos+neg))
print("  {:7,} # check against total_unique".format(total_unique))
print("\nthe {:,} positive patients have {:,} labelled boxes".format(pos, boxes))
import numpy as np
import matplotlib.lines as mlines

# max box height or width + 100, then rounded to nearest hundred
max_dim = np.round(np.ceil(np.max([df.width.max(), df.height.max()])) + 100, decimals=-2)

df.plot.scatter('width', 'height', title = 'Bounding Box Shapes', 
                xlim=(0, max_dim), ylim=(0, max_dim), s = 1)
plt.plot([0, max_dim], [0, max_dim], color = 'r', 
         linestyle="-", linewidth=1, label='Equal width & height')
plt.legend()
df.boxplot(['width', 'height'])