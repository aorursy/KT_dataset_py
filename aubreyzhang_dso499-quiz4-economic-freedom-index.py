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
economics = pd.read_csv('/kaggle/input/the-economic-freedom-index/economic_freedom_index2019_data.csv',index_col = 'Country',encoding='ISO-8859-1')

economics.head()

# economics is a dataframe created from csv file

# index_col is setting the dataframe's index to "Country" Column
economics.columns.to_list()

#to_list() change the output to list format
economics.shape
economics = economics[['Region','World Rank','Region Rank','2019 Score',

                        'Population (Millions)','GDP Growth Rate (%)'

                       ,'Unemployment (%)','Inflation (%)']]

#We can use a list of column names to select columns of a dataframe

economics.head()
economics.loc['Brazil','Unemployment (%)']

# first argument is the name of the row (index name), and the second argument is the column name

# By using both row and column names, we can locate the cell and return the cell value
# Step 1: select all the rows with GDP grotwh rate larger than or equal to 8.0

# Step 2: return the index of these rows (country names)

# Step 3: change the results in to list format

economics[economics['GDP Growth Rate (%)']>=8.0].index.to_list()
# Step 1: sort the table by using values in 2019 Score in descending order

# Step 2: use iloc[] to select the top 5 rows only

economics.sort_values('2019 Score',ascending=False).iloc[:5]
# Step 1: Split by Region

# Step 2: Select the relevant columns ('2019 Score' and 'GDP Growth Rate (%)')

# Step 2: use mean() to calculate average values of each economic indices

economics.groupby('Region')[['2019 Score','GDP Growth Rate (%)']].mean()
# Step 1: Split by Region

# Step 2: Select the column '2019 Score'

# Step 2: use idxmin() to find out the country that has minimum score in each region

economics.groupby('Region')['2019 Score'].idxmin()
economics.groupby('Region')['2019 Score'].std()
economics.groupby('Region').size()
# Step 1: Split by Region

# Step 2: Select columns '2019 Column'

# Step 3: Summarize by mean()

# Step 4: Sort by values, note that horizontal plot plots in reverse order, so we need to sort in ascending order to make plot in descending order

# Step 5: plot the horizontal bar plot

economics.groupby('Region')['2019 Score'].mean().sort_values().plot.barh()