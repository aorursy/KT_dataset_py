import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
train.head()
train.shape
test = pd.read_json('../input/stanford-covid-vaccine/test.json',lines=True)
test.head()
test.shape
bpps_files = os.listdir("../input/stanford-covid-vaccine/bpps/")
len(bpps_files)
import matplotlib.pyplot as plt

size = 4

fig, ax = plt.subplots(size, size, figsize=(10, 10))
ax = ax.flatten()
for i, bpps in enumerate(bpps_files):
    if i == size**2:
        break
    img = np.load(f"../input/stanford-covid-vaccine/bpps/{bpps}")
    ax[i].imshow(img)
    ax[i].set_title(bpps)
plt.show()
cols_to_predict = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
train[cols_to_predict]
reactivity = [round(t, 2) for t in train["reactivity"][0]][:12]
reactivity
deg_Mg_pH10 = [round(t, 2) for t in train["deg_Mg_pH10"][0]][:12]
deg_Mg_pH10
deg_Mg_50C = [round(t, 2) for t in train["deg_Mg_50C"][0]][:12]
deg_Mg_50C
pd.DataFrame({
     'reactivity':list(train.iloc[0].reactivity),
     'deg_Mg_pH10':list(train.iloc[0].deg_Mg_pH10),
     'deg_pH10':list(train.iloc[0].deg_pH10),
     'deg_Mg_50C':list(train.iloc[0].deg_Mg_50C),
     'deg_50C':list(train.iloc[0].deg_50C),
})
seq_str = pd.DataFrame({
     'sequence':list(train.iloc[0].sequence),
     'structure':list(train.iloc[0].structure),
     'predicted_loop_type':list(train.iloc[0].predicted_loop_type)
})
seq_str.head(15)
import seaborn as sns

sns.countplot(seq_str.sequence)
sns.countplot(seq_str.predicted_loop_type)
sns.countplot(seq_str.structure)
train.describe(include_objects=True)
rev = train.reactivity.unique()
rev = dict(zip(rev, range(len(rev))))
