import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
pkmn = pd.read_csv('../input/Pokemon.csv')

pkmn.head()
#Deletes columns Generation and Legendary

pkmn = pkmn.drop(['Generation', 'Legendary'],1)

pkmn.head()
#To start things off, let's just make a scatterplot based on two variables from the data set. I will use HP and Attack.



sns.jointplot(x="HP", y="Attack", data=pkmn,size=6,ratio=4,color='r');

plt.show()
sns.jointplot(x="Total",y="HP",data=pkmn,size=5,ratio=3,color='y')



plt.show()
#Now let's see if we can make something a little bit prettier.How about a distribution of all six stats? 

#We could even group it further using Pokemon type!

#Let's take it one step at a time.
sns.boxplot(data=pkmn,palette="bright");

plt.tight_layout() #Makes automatic editing.
pkmn = pkmn.drop(['Total','#'],1)

pkmn.head()


sns.boxplot(data=pkmn,palette="pastel",width=0.8)

plt.tight_layout()


pkmn = pd.melt(pkmn, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")

pkmn.head()


sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1",palette="Set1")

plt.figure(figsize=(12,10))

plt.ylim(0, 275)

sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(12,10))

plt.ylim(0, 275)

sns.violinplot(x="Stat", y="value", data=pkmn)

sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

plt.show()
sns.set_style("whitegrid")

with sns.color_palette([

    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    plt.figure(figsize=(12,10))

    plt.ylim(0, 275)

    sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=False, size=9)

    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);