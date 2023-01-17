import numpy as np
import pandas as pd
!pip install venndata
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df['Pclass1'] = df.Pclass.apply(lambda x: 1 if x==1 else 0)
df['Pclass2'] = df.Pclass.apply(lambda x: 1 if x==2 else 0)
df['Pclass3'] = df.Pclass.apply(lambda x: 1 if x==3 else 0)
df['male'] = df.Sex.apply(lambda x: 1 if x=='male' else 0)
df['female'] = df.Sex.apply(lambda x: 1 if x=='female' else 0)
df['senior'] = df.Age.apply(lambda x: 1 if x>60 else 0)
df2 = df[['Survived', 'Pclass1', 'Pclass2', 'Pclass3', 'male', 'female', 'senior']]

import matplotlib
matplotlib.rcParams['figure.figsize'] = [10, 10]
from venndata import venn
fineTune=False
labels, radii, actualOverlaps, disjointOverlaps = venn.df2areas(df2, fineTune=fineTune)
print(labels)
print(radii)
print(actualOverlaps)
print(disjointOverlaps)
fig, ax = venn.venn(radii, actualOverlaps, disjointOverlaps, labels=labels, labelsize='auto', cmap=None, fineTune=fineTune)