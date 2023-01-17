import pandas as pd

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')



# Import the data and do some cleaning

# This is a huge file, let's only get the header and the bottom of the file with 2017 data

#

donations = pd.read_excel("../input/Donations 2016 to 2018.xlsx")

# Ensure that the 'Total Amount' is a number and not a string. Rename the column to 'Dollars'

donations['Total Amount'] = pd.to_numeric(donations['Total Amount'])

donations.rename(columns={'Total Amount':'Dollars'}, inplace = True)



# Group the donations by Dollars and show the top 12.

totals = donations[['Political Party','Dollars']].groupby(['Political Party'], as_index=False).sum()

largest = totals.nlargest(12,'Dollars')



# Graph and show the results

ax = largest.plot(kind='bar', x='Political Party', title ="Politcal Contributions", figsize=(15, 8), legend=True, fontsize=12)

ax.set_xlabel("Political Party", fontsize=12)

ax.set_ylabel("Canadian Dollars", fontsize=12)

plt.show()

                                           

print(largest)