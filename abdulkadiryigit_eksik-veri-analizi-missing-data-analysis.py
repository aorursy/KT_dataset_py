#Missinno Kütüphanesi için

!pip install missingno
import pandas as pd

import numpy  as np

import seaborn as sns

import missingno as msno
hitters = pd.read_csv("../input/hitters/Hitters.csv")

df=hitters.copy()
df.info()
df[df.isnull().any(axis=1)]
df.isnull().sum()
df[df.isnull().any(axis=1)]
msno.bar(df)
msno.matrix(df)