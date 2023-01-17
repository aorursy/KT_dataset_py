import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/fifa-21-complete-player-dataset/fifa21_male2.csv")

cols=['Name', 'Age', 'OVA', 'Nationality', 'Club','BP','PAC','SHO', 'PAS', 'DRI','PHY','DEF']

df=df[cols]

df.head()
def radar_chart(players=['V. van Dijk','M. Salah'], title="Virgil van Dijk Vs Mo Salah"):

    """

    INPUT: 

    players: Player names(1D-array)

    title : Title for the chart(str)

    

    OUTPUT 

    Plots Radar Chart

    """

    labels=np.array(['PAC','SHO', 'PAS', 'DRI','PHY','DEF'])

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    angles=np.concatenate((angles,[angles[0]]))



    fig=plt.figure(figsize=(6,6))

    plt.suptitle(title)

    for player in players:

        stats=np.array(df[df.Name==player][labels])[0]

        stats=np.concatenate((stats,[stats[0]]))



        ax = fig.add_subplot(111, polar=True)

        ax.plot(angles, stats, 'o-', linewidth=2, label=player)

        ax.fill(angles, stats, alpha=0.25)

        ax.set_thetagrids(angles * 180/np.pi, labels)



    ax.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()

    

radar_chart()
radar_chart(players=['Cristiano Ronaldo','L. Messi'],

           title="Messi Vs Ronaldo")
df_90plus=df[df.OVA>90]

print("Number of Players with 90+ overall rating :{}".format(df_90plus.shape[0]))



radar_chart(players=df_90plus.Name,title="Top Rated Players")