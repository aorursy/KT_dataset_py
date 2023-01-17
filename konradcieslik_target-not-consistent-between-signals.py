import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))

df = pd.read_csv('../input/metadata_train.csv')
df.info()
df.head()
#let's check if targets are consistent within the same measurement id
targets = df.groupby('id_measurement')[['target','id_measurement']].agg('mean')
targets.head()
sns.countplot(x='target',data=targets)
# it should be only "1" and "0" but we have cases where target is not consitent 
mislabeled = targets.loc[(targets.target <1 ) & (targets.target > 0.3) ,'id_measurement']
print(str(mislabeled.shape[0]) + ' measurments most likely mislabeled' )

# qc it all

df.loc[df.id_measurement.isin(mislabeled) ,:]
