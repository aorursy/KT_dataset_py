import numpy as np

import pandas as pd

import scipy as sp



# Graph drawing

import matplotlib

from matplotlib import font_manager

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import rc



from matplotlib import animation as ani

from IPython.display import Image



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]



# pandas formatting

pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



%matplotlib inline

%config InlineBackend.figure_format='retina'
df = pd.read_csv("../input/nba-shot-logs/shot_logs.csv")
df.shape
df.head()
df.player_name.nunique()
df.player_name.value_counts()[:11]
df.SHOT_RESULT.value_counts()
df.SHOT_RESULT = df.SHOT_RESULT.map({"missed":0, "made":1})
df.SHOT_RESULT.value_counts()
player_name = df.player_name.value_counts().index[0]

player_name
success_mean = df[df.player_name==player_name].SHOT_RESULT.mean()

success_mean
shot_log = df[df.player_name==player_name].SHOT_RESULT.values
shot_log[:20]
set_size = 20
len(shot_log), (len(shot_log) // set_size)*set_size
shot_log = shot_log[:(len(shot_log) // set_size)*set_size].reshape(-1, set_size)

shot_log
shot_log_sum = shot_log.sum(axis=1)
plt.figure(figsize=(10,7))

plt.hist(shot_log_sum, bins=np.arange(shot_log_sum.min()-1, shot_log_sum.max()+1))

plt.title(f"{player_name}'s shot history per {set_size} shots.")
xx = np.arange(shot_log_sum.min()-1, shot_log_sum.max()+1)

p = sp.stats.binom.pmf(xx, set_size, success_mean)



plt.figure(figsize=(10,7))

plt.hist(shot_log_sum, bins=np.arange(shot_log_sum.min()-1, shot_log_sum.max()+1), density=True)

plt.plot(xx, p)

plt.title(f"{player_name}'s shot sum(set_size={set_size}) & binomial distribution.");
set_size = 20

for player_name in df.player_name.value_counts().index.tolist()[1:11]:

    success_mean = df[df.player_name==player_name].SHOT_RESULT.mean()

    shot_log = df[df.player_name==player_name].SHOT_RESULT.values

    shot_log = shot_log[:(len(shot_log) // set_size)*set_size].reshape(-1, set_size)

    shot_log_sum = shot_log.sum(axis=1)

    

    xx = np.arange(shot_log_sum.min()-1, shot_log_sum.max()+1)

    p = sp.stats.binom.pmf(xx, set_size, success_mean)



    plt.figure(figsize=(10,7))

    plt.hist(shot_log_sum, bins=np.arange(shot_log_sum.min()-1, shot_log_sum.max()+1), density=True)

    plt.plot(xx, p)

    plt.title(f"{player_name}'s shot sum(set_size={set_size}) & binomial distribution.");

    plt.show()