# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
crops_prod_data = pd.read_csv("../input/datafile.csv",index_col='Crop')
print(crops_prod_data.info())
print(crops_prod_data)
# drop all rows with Nan values
crops_prod_data = crops_prod_data.dropna()
print(crops_prod_data)
# plots for individual crops
crops_prod_data.loc['Rice',:].plot()
crops_prod_data.loc['Milk',:].plot()
crops_prod_data.loc['All Agriculture',:].plot()
plt.legend(loc='upper left')
# print index of dataframe
print(crops_prod_data.index)
# Transpose the dataset
print(crops_prod_data.T)
# Get current size of figure
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: current size
print("Current size:", fig_size)
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
# plot the transposed dataset
crops_prod_data.T.plot(subplots=True,layout=(3,4))
plt.xticks(rotation=45)
plt.tight_layout()

# now move to another dataset
cultivation_data = pd.read_csv("../input/datafile (1).csv")
print(cultivation_data.info())
print(cultivation_data.head())
print(cultivation_data.columns)
# Add additional columns in dataframe
cultivation_data['Per Hectare Cost Price'] = cultivation_data['Cost of Production (`/Quintal) C2'] * cultivation_data['Yield (Quintal/ Hectare) ']
cultivation_data['Cost of cultivation per hectare'] = cultivation_data['Cost of Cultivation (`/Hectare) A2+FL'] + cultivation_data['Cost of Cultivation (`/Hectare) C2']
cultivation_data['Yield in Kg per hectare'] = cultivation_data['Yield (Quintal/ Hectare) '] * 100
print(cultivation_data.head())
# checking transposed dataframe
print(cultivation_data.T.head())
# checking number of data points for different crops
print(cultivation_data.Crop.value_counts())
# checking columns of dataframe
print(cultivation_data.columns)
# creating list of specific columns from dataframe
columns = ['Crop','State','Yield (Quintal/ Hectare) ']
# creating subset of dataframe using list of specific columns
new_data = cultivation_data[columns]
# pivoting the dataframe
table = new_data.pivot('Crop','State','Yield (Quintal/ Hectare) ')
# fill Nan values with zero
table = table.fillna(0)
# plot (stacked bar plots) pivoted dataframe
table.plot(kind='bar',stacked=True,colormap='Paired')
# setting Y label
plt.ylabel('Yield (Quintal/ Hectare)')

# plotting transposed dataframe
table.T.plot(kind='bar',stacked=True)
plt.ylabel('Yield (Quintal/ Hectare) ')
# setting legend location to best
plt.legend(loc='best')
columns = ['Crop','State','Yield in Kg per hectare']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Yield in Kg per hectare')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)
labels = []
for j in table.T.columns:
    for i in table.T.index:
        # values are added as factor to fit into plot
        label = round((int(table.T.loc[i][j])),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2.,y + height/2.,label, ha='center',va='center')

plt.ylabel('Yield (Kg/ Hectare)')
# setting legend location to best
plt.legend(loc='best')
columns = ['Crop','State','Per Hectare Cost Price']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Per Hectare Cost Price')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)
labels = []
for j in table.T.columns:
    for i in table.T.index:
        # values are added as factor to fit into plot
        label = round((int(table.T.loc[i][j])/10000),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2.,y + height/2.,label, ha='center',va='center')

plt.ylabel('Per Hectare Cost Price')

columns = ['Crop','State','Cost of cultivation per hectare']
new_data = cultivation_data[columns]
table = new_data.pivot('Crop','State','Cost of cultivation per hectare')
table = table.fillna(0)
ax = table.T.plot(kind='bar',stacked=True)

labels = []
for j in table.T.columns:
    for i in table.T.index:
        label = round((int(table.T.loc[i][j])/10000),1)
        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):
    width = rect.get_width()
#    if width > 0:
    x = rect.get_x()
    y = rect.get_y()
    height = rect.get_height()
    ax.text(x + width/2.,y + height/2.,label,ha='center', va='center')
    
plt.ylabel('Cost of cultivation per hectare')

# now move to another dataset
crop_data = pd.read_csv("../input/datafile (2).csv")
print(crop_data.info())
print(crop_data.columns)
crop_data['Crop'] = crop_data['Crop             ']
del crop_data['Crop             ']
print(crop_data.columns)
columns = ['Crop','Production 2006-07', 'Production 2007-08', 'Production 2008-09','Production 2009-10', 'Production 2010-11']
columns2 = ['Crop','Area 2006-07', 'Area 2007-08', 'Area 2008-09','Area 2009-10', 'Area 2010-11']
columns3 = ['Crop','Yield 2006-07', 'Yield 2007-08', 'Yield 2008-09','Yield 2009-10', 'Yield 2010-11']
production = crop_data[columns]
area = crop_data[columns2]
yd = crop_data[columns3]
print(production.head())
production.index = production['Crop']
area.index = area['Crop']
yd.index = yd['Crop']
del production['Crop']
del area['Crop']
del yd['Crop']
# printing columns for transposed production dataframe
print(production.T.columns)
# Get current size of figure
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: current size
print("Current size:", fig_size)
# Set figure width to 12 and height to 9
fig_size[0] = 18
fig_size[1] = 18
plt.rcParams["figure.figsize"] = fig_size
# initialize i for loop
i=0
ax = production.T.plot(subplots=True,layout=(11,5),color='red',label='Production',legend=False)
# for loop on axes for setting titles for subplots
for a in ax.flat:
    a.set_title(production.index[i])
    i += 1
# plotting another dataframes on same plot
ax1 = area.T.plot(subplots=True,layout=(11,5),ax=ax, linestyle=':',marker='.',color='blue',legend=False)
ax2 = yd.T.plot(subplots=True,layout=(11,5),ax=ax, linestyle='--',marker="*",color='green',legend=False)
labels = ['Production','Area','Yield']
# setting X-axis labels
plt.xticks(np.arange(5),('2006-07','2007-08','2008-09','2009-10','2010-11'))
# placing the legend on the bottom of plot
plt.legend(labels=labels,loc='upper center', bbox_to_anchor=(-0.5, -0.5),  shadow=True, ncol=3, fontsize = 'xx-large')

plt.tight_layout()
