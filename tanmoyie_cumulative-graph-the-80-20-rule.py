# import the libraries

import pandas # Data import

import pylab as pl # Data visualization



# import the dataset

maintenance_df = pandas.read_excel("../input/maintenance-time-of-machines/maintenance time of machines.xlsx") # Line 1

maintenance_df.head()
# Draw the Histogram

maintenance_time = maintenance_df.iloc[:,1]

N, bins, patches = pl.hist(maintenance_time, bins=16)

pl.title("Histogram")

pl.xlabel("Time of Maintenance")

pl.ylabel("Frequency")

jet = pl.get_cmap('jet', len(patches))

for i in range(len(patches)):

    patches[i].set_facecolor(jet(i))
# Develop a Cumulative graph

N, bins, patches = pl.hist(maintenance_time, cumulative = True, bins=16)

pl.xlabel("Time of Maintenance")

pl.ylabel("Cumulative Probability Distribution")

jet = pl.get_cmap('jet', len(patches))

for i in range(len(patches)):

    patches[i].set_facecolor(jet(i))