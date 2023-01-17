import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



age_group = pd.read_csv("../input/population_by_age_group.csv", index_col='Index')
age_group.info()

age_group.head()

# I hid the output to make the viewing easier but you can view it
# For this plot I only need '0-4' column so I will only change that data column.



# Removing whitespace in between numbers of '0-4' column

age_group['0-4'] = age_group['0-4'].str.replace(" ","")



# Making '0-4' column data from object to int64

age_group['0-4'] = age_group['0-4'].astype(str).astype(int)
age_group.head()
#Making new dataframe with only required columns will make it easier to view and plot data.



indices = age_group['Region, subregion, country or area *'] == 'WORLD'

world = age_group.loc[indices,:] 



indices = age_group['Region, subregion, country or area *'] == 'Sub-Saharan Africa'

sub_saharan_africa = age_group.loc[indices,:]
#making rest of the world dataframe to make it like world and sub_saharan_africa

rest_of_world_diff = np.subtract(world['0-4'],sub_saharan_africa['0-4'])

rest_of_world = pd.DataFrame( columns=['Reference date (as of 1 July)', '0-4'])

rest_of_world['0-4'] = rest_of_world_diff

rest_of_world['Reference date (as of 1 July)'] = np.arange(2015, 2105, 5)

print(rest_of_world) # I hid the output but you can view it
xticks = np.arange(2015, 2101, 5)

yticks = np.arange(100000, 600001, 100000)



plt.figure(figsize=(16, 9), dpi=200)

plt.plot(sub_saharan_africa['Reference date (as of 1 July)'], sub_saharan_africa['0-4'], color='#4c81a3')

plt.plot(rest_of_world['Reference date (as of 1 July)'], rest_of_world['0-4'], color='#fd8538')

plt.xlabel('Year')

plt.ylabel('Kids aged 0 - 4')

plt.xticks(xticks)

plt.yticks(yticks,['100M','200M','300M','400M','500M','600M'])

plt.grid(color='r', linestyle='-', alpha=0.1)

plt.title('Kids aged 0 -4 ')

plt.text(2057, 300000, 'SUB-SAHARAN AFRICA',rotation=10, horizontalalignment='center',verticalalignment='top',multialignment='center', color='#4c81a3', fontsize='15')

plt.text(2057, 470000, 'THE REST OF THE WORLD',rotation=-10, horizontalalignment='center',verticalalignment='top',multialignment='center', color='#fd8538', fontsize='15')

plt.text(2020, 170269, '173M', fontweight='bold',rotation=20, fontsize='15', color='#4c81a3')

plt.text(2095, 275531, '293M', fontweight='bold',rotation=0, fontsize='15', color='#4c81a3')

plt.text(2020, 509344, '506M', fontweight='bold',rotation=-10, fontsize='15', color='#fd8538')

plt.text(2095, 369356, '357M', fontweight='bold',rotation=-10, fontsize='15', color='#fd8538')

plt.show()