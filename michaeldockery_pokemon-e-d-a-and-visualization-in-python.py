#This is a practice with EDA Data Visualization and Pandas.
#A lot of this code is referencing :https://www.kaggle.com/yassinealouini/pokemon-eda#%C2%A0Load-the-data
#I am  a beginner with kaggle and am still learning. Some feedback is well appreciated

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp

#upload the data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename));
Pokedex=pd.read_csv('/kaggle/input/pokemon/Pokemon.csv');
sns.boxplot(data=Pokedex);
Pokedex.head()
pkmn = Pokedex.drop(['Total', '#',"Legendary","Generation"],1)
sns.boxplot(data=pkmn);
profile=pdp.ProfileReport(pkmn)
profile.to_notebook_iframe();
pkmn_melt = pd.melt(pkmn, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")

pkmn_melt.head()
sns.swarmplot(x="Stat", y="value", data=pkmn_melt, hue="Type 1");
plt.figure(figsize=(12,10))
plt.ylim(0, 275)
sns.swarmplot(x="Stat", y="value", data=pkmn_melt, hue="Type 1", dodge=True, size=7)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);