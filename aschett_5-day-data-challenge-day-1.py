# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read the csv match data into a DataFrame

lol = pd.read_csv('../input/LeagueofLegends.csv')
# Check the size of the DataFrame

lol.shape
# Peek at the start of the DataFrame

lol.head() # lol.head(10)

# or the bottom: lol.tail()
# Show the index. In this case, an integer index

lol.index
# Show the column headings

lol.columns

# You can show the columns too: lol.values
# Print a general analysis

lol.describe()
# Another general summary including indices, column types, empty entries, and data usage

lol.info()
# Transpose!

lol.T
# Sort by axis labels

lol.sort_index(axis=1, ascending=False)

# Selecting axis=1 refers to sorting the columns. Axis 0 is the row labels
# Sort by the data in a column

lol.sort_values(by=['gamelength', 'bResult'], ascending=[False, True]) # :0
# Select a single column...with unique entries

lol['blueTopChamp'].drop_duplicates()

# This type is a Series
# Slice the row indices

lol[5:15]
# Get a row by label...ours is just a integer

lol.loc[3]
# Slice along index and show multiple columns

lol.loc[1:9,['blueTop','gamelength']]

# Slicing the index includes both ends
# Get an individual scalar (faster than loc)

lol.at[3,'redJungleChamp']
# Slice row and column using indices insted of labels

lol.iloc[5:10,3:9]

# Also lol.iat[3, 13]
# Get all rows matching a column condition

lol[lol.blueMiddleChamp=='Lux']
# Use .isin for alternatives                # Show the counts of each value in the column

lol[lol.blueJungleChamp.isin(['LeeSin', 'Amumu'])]['blueJungleChamp'].value_counts()