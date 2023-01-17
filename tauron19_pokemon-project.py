# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualisation
sn.set(color_codes = True, style = "white")
import matplotlib.pyplot as ml # data visualisation
import warnings
warnings.filterwarnings("ignore")
poke = pd.read_csv("../input/Pokemon.csv", sep=",", header = 0)
print(poke.shape)


# Any results you write to the current directory are saved as output.
poke.head(10)
poke .corr() 
sn.jointplot(x="Attack", y="Defense", data=poke);

co = poke.corr()
sn.heatmap(co)
sn.violinplot(x="Legendary", y="Total", data= poke, size=10)
poke = poke.drop(['Generation', 'Legendary'],1)
poke = poke.drop(['Total', '#'],1)
sn.boxplot(data=poke);

poke = pd.melt(poke, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")
ml.figure(figsize=(12,10))
ml.ylim(0, 275)
sn.swarmplot(x="Stat", y="value", data=poke, hue="Type 1", split=True, size=7)
ml.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);