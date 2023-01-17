import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy as np

from matplotlib import pyplot
m = pd.read_csv('../input/Mountains.csv')
m.head(5)
m.shape
m.rename(columns={'Height (m)' : 'Height(m)', 'Height (ft)' : 'Height(ft)', 'Prominence (m)' : 'Prominence(m)', \

                  'Parent mountain' : 'Parent', 'First ascent' : 'First_Ascent', 'Ascents bef. 2004' : 'Ascents', \

                  'Failed attempts bef. 2004' : 'Failed_Attempts'}, inplace = True)

m['Failed_Attempts'] = m['Failed_Attempts'].fillna(0).astype(int)

m.set_value(0, "Ascents", "145")

m['Parent'] = m['Parent'].fillna('None').astype(object)



#Since we cannot quantify 'many' Ascents Muztagh Ata is removed from the data set

m = m[m.Ascents != 'Many']



m['Ascents'] = m['Ascents'].fillna("0")

m['Ascents'] = m['Ascents'].astype(int)



#No partial failures! :|

m['Failed_Attempts'] = m['Failed_Attempts'].astype(int)
"""For any analysis that covers the 'First Ascent' variable, "climbed" is a second set of data that will be used only 

including mountains that have been climbed"""

climbed = m[m.First_Ascent != 'unclimbed']

climbed['First_Ascent'] = climbed['First_Ascent'].astype(int)
"""There are only four mountains in the data set that are going to be removed from the original data set 'm' in 'climbed'

"""

m[m.First_Ascent == 'unclimbed']
#Confirming all records are non-null and are the correct type

climbed.info()
mtn, hi = pyplot.subplots(figsize=(6,6))

mclimbed = climbed[["Rank", "Height(m)", "Prominence(m)", "First_Ascent", "Ascents", "Failed_Attempts"]]

color = plt.cm.terrain

plt.title("Correlation of Mountain Data", size = 20, y = 1.07)

sns.heatmap(mclimbed.astype(float).corr(), linewidths = 0.3,vmax = 1.0, square = True, \

            cmap = color, linecolor = 'white', annot = True)

sns.plt.show()
p = sns.pairplot(mclimbed, kind = 'reg')

sns.plt.show()
mtn, hi = pyplot.subplots(figsize=(10,10))

color = 'seismic'

success_rate = (m['Ascents'] / ((m['Ascents']) + (m['Failed_Attempts'])))*100

success_rate.fillna(0).astype(float)

#MM(Modified Mountains)

mm = m

mm['success_rate'] = success_rate

mm['success_rate'].round(2)

vm = mm.pivot("Height(m)", "Prominence(m)", "success_rate")

cPreference = sns.heatmap(vm, vmax = 100, cmap = color, xticklabels = 10, yticklabels = 5, cbar_kws={'label': 'Success Rate of Climbs (%)'})

cPreference = cPreference.invert_yaxis()

plt.title("What is a good Mountain to Climb?", size = 20)

sns.plt.show()
color = 'jet'

plt.figure(figsize=(10,10))

plt.title("Where do people like to Climb?", size = 20)

vm = mm.pivot("Height(m)", "Prominence(m)", "Ascents")

sns.set_style("ticks")

mPreference = sns.heatmap(vm, vmax = 145, cmap = color, xticklabels = 10, yticklabels = 5,cbar_kws={'label': 'Number of Successful Climbs'})

mPreference = mPreference.invert_yaxis()

sns.plt.show()
sns.regplot(y = success_rate, x = m['Height(m)'], color = 'green')

plt.ylabel("Success Rate (%)")

sns.plt.show()

sns.regplot(y = success_rate, x = m['Rank'], color = 'green')

plt.ylabel("Success Rate (%)")

sns.plt.show()
sns.regplot(x = 'First_Ascent', y = 'Height(ft)', data = climbed)



"""Line Represents the year where the most First ascents occured"""

x=climbed['First_Ascent'].mode()

plt.plot([x,x], [23000,29000], linewidth = 2, color = "green")



sns.plt.show()



sns.regplot(x = 'Ascents', y = 'Height(ft)', data = m)

sns.plt.show()