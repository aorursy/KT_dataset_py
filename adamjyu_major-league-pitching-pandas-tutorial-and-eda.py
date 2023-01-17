# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 1. Use pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv') to create a DataFrame called pitching.

# 2. Use .head() to display the first 5 rows of the dataset.



pitching = pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv')

pitching.head()
# 1. Use conditional selecting to only select rows where the `yearID` is greater than or equal to 1990, `GS` is greater than or equal to 26, and `G` is equal to `GS`. 

# 2. Use .iloc[:, :20] to keep every row in which the three conditions are true and keep only the first 20 columns using slicing. The first parameter `:` means every row in which the three conditions are true will be kept. `:20` means that only the first 20 columns are kept.

# 3. Use .drop() to eliminate specific columns from the dataset. Within .drop() input a list of column names that you wish to drop and set `axis=1` to drop columns rather than rows. If we had set `axis=0`, we would have to include index names of the entire rows we wanted to drop.

# 4. Use .head() to display the first 5 rows of the new dataset.

# Remember to save the dataset as a variable name for each step.



pitching = pitching[(pitching['yearID'] >= 1990) & (pitching['GS'] >= 26) & (pitching['G'] == pitching['GS'])].iloc[:, :20]

pitching = pitching.drop(['stint', 'W', 'L', 'CG', 'SHO', 'SV'], axis=1)

pitching.head()
# 1. Use set_index('playerID') to make each `playerID` the index. Save the dataset as a variable.

# 2. Use .head(7) to display the first 7 rows of the updated dataset. In Pandas, you can choose the number of rows to display using .head(). As stated above, the default value is 5.



pitching = pitching.set_index('playerID')

pitching.head(7)
# Use .tail(6) to show the bottom 6 rows of the dataset. Much like .head(), you can choose the number of bottom rows to display. Also like .head(), the default value for .tail() is 5.

pitching.tail(6)
# Use .shape[0] to determine the number of rows in the dataset. Using .shape[1] would have returned the number of columns and .shape would have returned (rows, columns).



pitching.shape[0]
# 1. To create a new column, index the new column name like you would do for a dictionary.

# 2. Perform the calculation of selecting the `IPouts` column and dividing by 3.

# 3. Use .round(2) to round the results to 2 decimal places.

# 4. Use the del function and select the `IPouts` column to delete the column from the dataset.

# 5. Use .head() to display the first 5 rows of the updated dataset.



pitching['IP'] = ((pitching['IPouts'])/3).round(2)

del pitching['IPouts']

pitching.head()
# 1. To create a new column, index the new column name like you would do for a dictionary.

# 2. Perform the calculation of selecting the `SO` column, multiplying it by 9, and dividing by the `IP` column.

# 3. Use .round(2) to round the results to 2 decimal places.



pitching['SO9'] = (pitching['SO'] * 9 / pitching['IP']).round(2)
# 1. To create a new column, index the new column name like you would do for a dictionary.

# 2. Perform the calculation of selecting the `BB` column,  multiplying it by 9, and then dividing by the `IP` column.

# 3. Use .round(2) to round the results to 2 decimal places.

# 4. Use .head() to display the first 5 rows of the updated dataset.



pitching['BB9'] = (pitching['BB'] * 9 / pitching['IP']).round(2)

pitching.head()
# 1. To create a new column, index the new column name like you would do for a dictionary.

# 2. Perform the calculation of selecting the `SO` column and dividing by the `BB` column.

# 3. Use .round(2) to round the results to 2 decimal places.

# 4. Use .head() to display the first 5 rows of the updated dataset.



pitching['SOtoBB'] = (pitching['SO'] / pitching['BB']).round(2)

pitching.head()
# 1. Split by the `yearID` column.

# 2. Select the `IP` column.

# 3. Summarize with `.sum()`.

# 4. Divide

# 5. Split by the `yearID` column.

# 6. Select the `GS` column.

# 7. Summarize with `.sum()`.

# 4. Plot bar chart



(pitching.groupby('yearID').IP.sum()/pitching.groupby('yearID').GS.sum()).plot.bar()
# 1. Split by the `yearID` column.

# 2. Use .size() method to count the number of pitchers for each season.

# 3. Plot bar chart



pitching.groupby('yearID').size().plot.bar()
# 1. Split by the `yearID` column.

# 2. Select the `SO9` column.

# 3. Summarize with `.mean()`.

# 4. Plot bar chart



pitching.groupby('yearID').SO9.mean().plot.bar()
# 1. Select the SO column using pitching.SO

# 2. Use idxmax() to find the index corresponding to the maximum value.



pitching.SO.idxmax()
# 1. Select the SO column using pitching.SO

# 2. Use idxmin() to find the index corresponding to the minimum value.



pitching.SO.idxmin()
# 1. Select the ERA column using pitching.ERA

# 2. Use sort_values() to sort the values from smallest to largest.

# 3. Use .head(10) to display the smallest ERA to the 10th smallest ERA along with the playerIDs.



pitching.ERA.sort_values().head(10)
# 1. Select the SO9 column using pitching.SO9

# 2. Use sort_values(ascending=False) to sort the values from largest to smallest. The ascending=False parameter makes the values go from largets to smallest. The default value is ascending=True, meaning the values are sorted from smallest to largest.

# 3. Use .head(10) to display the largest SO9 to the 10th largest SO9 along with the playerIDs.



pitching.SO9.sort_values(ascending=False).head(10)
# 1. Select the BB9 column using pitching.BB9

# 2. Use sort_values() to sort the values from smallest to largest.

# 3. Use .head(10) to display the smallest BB9 to the 10th smallest BB9 along with the playerIDs.





pitching.BB9.sort_values().head(10)
# 1. Select the SOtoBB column using pitching.SOtoBB

# 2. Use sort_values(ascending=False) to sort the values from largest to smallest.

# 3. Use .head(10) to display the largest SOtoBB to the 10th largest SOtoBB along with the playerIDs.



pitching.SOtoBB.sort_values(ascending=False).head(10)