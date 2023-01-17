import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/voicegender/voice.csv')
import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt



%matplotlib inline
df.info()
df.head()
f, axis = plt.subplots(figsize = (18,18))

sns.heatmap(df.corr(), annot = False, linewidths = .4, ax = axis)

plt.show()
df.label.value_counts()
df.label = [1 if each == 'female' else 0 for each in df.label]
y = df['label'].values

x = df.drop(['label'], axis = 1)
y
x.head()
F