import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
lb = pd.read_csv('../input/champslb20190828/publicleaderboarddata (1)/champs-scalar-coupling-publicleaderboard.csv')
fig, ax = plt.subplots(1,1, figsize=(16,4))

scores = lb[['TeamId','Score']].groupby('TeamId').min().sort_values(by='Score').Score

#sns.distplot(scores, bins = np.linspace(-3.5, 2, 91), kde=False, ax=ax)

ax.hist(scores, bins = np.linspace(-3.3, 2, 111))

xl = plt.xlim([-3.3, 2],)

ax.set_ylim(0,125)

ax.set_xticks(np.linspace(-3.3, 2, 26))

ax.set_xticklabels([ f"{v:.1f}" for v in np.linspace(-3.3, 2, 26)])



public_kernels = [

    ("@artgor EDA", 0.289),

    ("@artgor Brute Force", -0.595),

    ("@todnewman NN", -1.073),

    #("@giba", -1.17),

    ("@fnand MPNN", -1.28),

    ("@xwxw2929 NN", -1.672),

    ("@iooohoooi Stack", -1.829),

    ("Public leak", -2.16),

]



bbox  = dict(boxstyle="round", fc="0.8")

arrow = dict(arrowstyle="->",connectionstyle = "angle,angleA=45,angleB=90,rad=10",

    edgecolor="darkgray",linewidth=2)



for who, score in public_kernels:

    ax.annotate(who, xy=(score, 0.1), xytext=(score, 130), rotation=45, bbox=bbox,arrowprops=arrow)