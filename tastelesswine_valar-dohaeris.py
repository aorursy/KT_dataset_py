import pandas as pd

import glob

import seaborn as sns

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline
battles=pd.read_csv('../input/battles.csv')

cd=pd.read_csv('../input/character-deaths.csv')

cp=pd.read_csv('../input/character-predictions.csv')
battles.head(50)
battles.groupby(['attacker_king','defender_king']).count()['name'].plot(kind='barh')
battles.groupby(['attacker_king','attacker_outcome']).count()['name'].unstack().plot(kind='barh')
battles.groupby(['attacker_king','battle_type']).count()['name'].unstack().plot(kind='barh')
battles.groupby(['region']).count()['name'].plot(kind='barh')
battles.groupby(['year']).count()['name'].plot(kind='barh')
p = battles.groupby('year').sum()[["major_death", "major_capture"]].plot.bar(rot = 0)

_ = p.set(xlabel = "Year", ylabel = "No. of Death/Capture Events", ylim = (0, 9)), p.legend(["Major Deaths", "Major Captures"])