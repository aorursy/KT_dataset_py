import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_pickle("../input/death-and-population-in-france-19902019/preprocessed_data/INSEE_deces_2010_2019.pkl")
df = df[df['date_deces']>='2010-01-01']
df = df.groupby(["year","weeknumber", "age_bin", "sexe", "departement_deces"], as_index=False)["nb_deces"].sum()
df.head()
df['nb_deces'] = df['nb_deces'].astype('float') # matplotlib struggles with pandas int
df_nat = df.groupby(["year","weeknumber"], as_index=False)["nb_deces"].sum()
# Weeknumber 1, 52, 53 are noisy (why?) so I remove them for this plot

df_nat = df_nat[(df_nat.weeknumber>1) & (df_nat.weeknumber<52)] 
ax = plt.subplot(1,1,1)

for y in np.arange(2010,2018):

    ax.plot(df_nat.weeknumber[df_nat.year==y], df_nat.nb_deces[df_nat.year==y].astype('float'), '-', color="0.8");

p2, = ax.plot(df_nat.weeknumber[df_nat.year==2018], df_nat.nb_deces[df_nat.year==2018].astype('float'), '-', label="2018");

p3, = ax.plot(df_nat.weeknumber[df_nat.year==2019], df_nat.nb_deces[df_nat.year==2019].astype('float'), '-', label="2019");

ax.legend()

plt.title("Death records in France according to INSEE")

plt.xlabel("Week number")

plt.ylabel("Number of deaths")

plt.show()