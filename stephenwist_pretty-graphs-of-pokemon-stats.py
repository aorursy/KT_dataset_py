import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline



pkmn = pd.read_csv('../input/Pokemon.csv')

pkmn.head()
pkmn["Type 2"] = pkmn["Type 2"].fillna("")

pkmn["Type"] = pkmn["Type 1"] + pkmn["Type 2"] 

pkmn["Name"] = pkmn["Name"].str.replace(".*(?=Mega)", "")

pkmn = pkmn.set_index(pkmn["Name"])

pkmn = pkmn.drop(["#", "Name","Type 1", "Type 2"], axis=1)

pkmn.loc["Mew", "Legendary"] = True

pkmn.head()
# make separate dataframe for all legendaries



legends = pkmn[pkmn["Legendary"] == True]

pkmn = pkmn.drop(legends.index[:])

legends.head()
gen_totals_no_legendaries = list()

for i in [1,2,3,4,5,6]:

    gen = pd.DataFrame(pkmn.loc[x] for x in pkmn.index if pkmn.loc[x]["Generation"] == i)

    gen_totals_no_legendaries.append(int(sum(gen["Total"])))



gen_totals_legendaries = list()

for i in [1,2,3,4,5,6]:

    gen = pd.DataFrame(legends.loc[x] for x in legends.index if legends.loc[x]["Generation"] == i)

    gen_totals_legendaries.append(int(sum(gen["Total"])))



plt.figure(1, figsize=(10, 10))

plt.subplot(311)

plt.ylabel('Total')

plt.title('No Legendaries')

sb.barplot([1,2,3,4,5,6], gen_totals_no_legendaries)



plt.subplot(312)

plt.ylabel('Total')

plt.title("Only Legendaries")

sb.barplot([1,2,3,4,5,6], gen_totals_legendaries)



plt.subplot(313)

plt.ylabel("Total")

plt.xlabel('Generation')

plt.title("Both")

sb.barplot([1,2,3,4,5,6], list(map((lambda x, y: x+y), \

                                gen_totals_no_legendaries, gen_totals_legendaries)) )

plt.figure(1, figsize=(12,12))



plt.subplot(211)

plt.title("No Legendaries")

sb.violinplot(x=pkmn["Generation"], y=pkmn["Total"])

sb.stripplot(x=pkmn["Generation"], y=pkmn["Total"], jitter=True, linewidth=1)



plt.subplot(212)

plt.title("Only Legendaries")

sb.violinplot(x=legends["Generation"], y=legends["Total"])

sb.stripplot(x=legends["Generation"], y=legends["Total"], jitter=True, linewidth=1)

types = list()

for type_ in pkmn["Type"].values:

    if type_ not in types:

        types.append(type_)

types = sorted(types)



print(types[:36])
# may the coding gods forgive me, for I am about to sin



types1 = pd.DataFrame()

types2 = pd.DataFrame()

types3 = pd.DataFrame()

types4 = pd.DataFrame()

for type_ in types[:36]:

    types1 = types1.append(pkmn[pkmn["Type"] == type_])

    

for type_ in types[36:71]:

    types2 = types2.append(pkmn[pkmn["Type"] == type_])



for type_ in types[71:102]:

    types3 = types3.append(pkmn[pkmn["Type"] == type_])



for type_ in types[102:]:

    types4 = types4.append(pkmn[pkmn["Type"] == type_])



types1
sb.plotting_context()
pkmn_type = pkmn.sort_values(by="Type")



sb.set_context(context='notebook', font_scale=2)



fig = plt.figure(1, figsize=(15, 65))



ax = fig.add_subplot(411)

sb.swarmplot(x="Type", y="Total", hue="Generation", data=types1, size=9, split=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)



ax = fig.add_subplot(412)

sb.swarmplot(x="Type", y="Total", hue="Generation", data=types2, size=9, split=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)



ax = fig.add_subplot(413)

sb.swarmplot(x="Type", y="Total", hue="Generation", data=types3, size=9, split=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)



ax = fig.add_subplot(414)

sb.swarmplot(x="Type", y="Total", hue="Generation", data=types4, size=9, split=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=80)
