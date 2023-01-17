import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
sn.set(color_codes = True, style="white")
import matplotlib.pyplot as ml
import warnings
warnings.filterwarnings("ignore")
poke = pd.read_csv("../input/Pokemon.csv", sep = ",", header = 0)
print (poke.shape, "rows and columns")


print(poke.head(10))
print(poke.corr())
sn.heatmap(poke.corr())