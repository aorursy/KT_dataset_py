import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats
import matplotlib.pyplot as plt

files = []

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        
# Let's import the dataset and describe it.
data = pd.read_csv(files[0])
data.head()
data.describe()
# Final results (in Italy)
dec = 4
print('TURNOUT:', np.around(data['VOTANTI'].sum() * 100 / data['ELETTORI'].sum(), decimals=dec), '%')
print('YES:', np.around(data['NUMVOTISI'].sum() * 100 / data['VOTANTI'].sum(), decimals=dec), '%')
print('NO:', np.around(data['NUMVOTINO'].sum() * 100 / data['VOTANTI'].sum(), decimals=dec), '%')
print('BLANK:', np.around(data['SCHEDEBIANCHE'].sum() * 100 / data['VOTANTI'].sum(), decimals=dec), '%')
print('INVALID:', np.around(data['SCHEDENULLE'].sum() * 100 / data['VOTANTI'].sum(), decimals=dec), '%')
print('DISPUTED:', np.around(data['SCHEDECONTESTATE'].sum() * 100 / data['VOTANTI'].sum(), decimals=dec), '%')
# I want some relative values
data['AFFLUENZA'] = data['VOTANTI'] / data['ELETTORI']
data['RELVOTISI'] = data['NUMVOTISI'] / data['VOTANTI'] 
data['RELVOTINO'] = data['NUMVOTINO'] / data['VOTANTI'] 
data['RELSCHEDEBIANCHE'] = data['SCHEDEBIANCHE'] / data['VOTANTI'] 
data['RELSCHEDENULLE'] = data['SCHEDENULLE'] / data['VOTANTI'] 
data['RELSCHEDECONTESTATE'] = data['SCHEDECONTESTATE'] / data['VOTANTI'] 
data.describe()
# Top ten YES municipalities
data.nlargest(10, 'RELVOTISI')[['REGIONE', 'PROVINCIA', 'COMUNE', 'ELETTORI', 'VOTANTI', 'RELVOTISI']]
# Top ten NO municipalities
data.nlargest(10, 'RELVOTINO')[['REGIONE', 'PROVINCIA', 'COMUNE', 'ELETTORI', 'VOTANTI', 'RELVOTINO']]
# Let's see if there's some correlation between population (linked to number of voters) and the results

# slope, intersept, and correlation coefficient calculation 
slope, intercept, r, p, stderr = scipy.stats.linregress(data['RELVOTISI'], data['ELETTORI'])

line_label = "R = " + str(np.around(r, decimals=3))

# plotting
fig, ax = plt.subplots(figsize = (14,8))
ax.plot(data['RELVOTISI'], data['ELETTORI'], linewidth=0, marker='o', markersize=2)
ax.plot(data['RELVOTISI'], intercept + slope * data['RELVOTISI'], label = line_label)
ax.set_xlabel('RELVOTISI')
ax.set_ylabel('ELETTORI')
ax.set_ylim(bottom=0, top=100000)
ax.legend(facecolor='white', fontsize=20)
plt.show()
# What about the correlation between voter turnout and the results?
# Let's see if there's some correlation between population (linked to number of voters) and the results

# slope, intersept, and correlation coefficient calculation 
slope, intercept, r, p, stderr = scipy.stats.linregress(data['RELVOTISI'], data['AFFLUENZA'])

line_label = "R = " + str(np.around(r, decimals=3))

# plotting
fig, ax = plt.subplots(figsize = (14,8))
ax.plot(data['RELVOTISI'], data['AFFLUENZA'], linewidth=0, marker='o', markersize=2)
ax.plot(data['RELVOTISI'], intercept + slope * data['RELVOTISI'], label = line_label)
ax.set_xlabel('RELVOTISI')
ax.set_ylabel('AFFLUENZA')
ax.legend(facecolor='white', fontsize=20)
plt.show()
# Aggregate
regional_data = data.groupby(['REGIONE']).sum()

# Fix relative values
regional_data['AFFLUENZA'] = regional_data['VOTANTI'] / regional_data['ELETTORI']
regional_data['RELVOTISI'] = regional_data['NUMVOTISI'] / regional_data['VOTANTI'] 
regional_data['RELVOTINO'] = regional_data['NUMVOTINO'] / regional_data['VOTANTI'] 
regional_data['RELSCHEDEBIANCHE'] = regional_data['SCHEDEBIANCHE'] / regional_data['VOTANTI'] 
regional_data['RELSCHEDENULLE'] = regional_data['SCHEDENULLE'] / regional_data['VOTANTI'] 
regional_data['RELSCHEDECONTESTATE'] = regional_data['SCHEDECONTESTATE'] / regional_data['VOTANTI'] 

regional_data
# Top ten YES regions
regional_data.nlargest(10, 'RELVOTISI')[['ELETTORI', 'VOTANTI', 'RELVOTISI']]
# Top ten NO regions
regional_data.nlargest(10, 'RELVOTINO')[['ELETTORI', 'VOTANTI', 'RELVOTINO']]
