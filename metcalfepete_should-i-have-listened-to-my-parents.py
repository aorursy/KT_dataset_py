import pandas as pd

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')



# Import the data and do some cleaning

# This is a huge file, let's only get the header and the bottom of the file with 2017 data

#

sunshine = pd.read_csv('../input/sunshine_list_2018-2016.txt', encoding = "ISO-8859-1", sep='\t', lineterminator='\n', skiprows=[i for i in range(1,870000)] )



#Do a little bit of renaming and clean-up

sunshine.rename(columns={'Calendar Year':'Year', 'Salary Paid':'Salary'}, inplace = True)

sunshine = sunshine[pd.notnull(sunshine.Salary)]



# Get only 2017 data and just the sector and salary columns 

job2017 = sunshine.query('Year == "2017"')[['Salary','Sector']]



# Print out the average and max salary for each government sector

print("Average Salary by Sectors\n", job2017.groupby(['Sector']).mean() )

print("\n\n","Max Salary by Sectors\n", job2017.groupby(['Sector']).max() )



# Show the salary ranges for each sector

# Make the sector names shorter for the plot

job2017.Sector = job2017.Sector.str.replace('Government of Ontario - ','')

job2017.boxplot( by = ['Sector'], rot=30, fontsize=12, showfliers=False, showmeans=True,  figsize=(15, 8))

plt.show()