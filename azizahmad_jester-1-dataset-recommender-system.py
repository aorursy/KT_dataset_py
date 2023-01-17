# import data analysis and visualization libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
# read the jester excel file, give the columns a representative number since the column names for the jokes was not provided

data = pd.read_excel('../input/jester-data-1.xls',names=(x for x in range(0,101)))
# rename first column for clarity

data.rename(columns={0:'num of ratings'},inplace=True)
data

# null values were indicated as 99.00
data.mean().sort_values(ascending=False).head(5)

# top rated jokes not cleaned from the 99.00 values
data.replace(to_replace=99,value=np.nan,inplace=True)

# changing the 99.00 null values to NaN
data.mean().sort_values(ascending=False).head(5)

# accurate list of top rated jokes
# let's explore the top rated joke, #50

data[50].value_counts().head(10)
# Visualize the ratings for joke # 50

plt.figure(figsize=(10,6))

plt.hist(data[50],bins=30)

plt.xlabel('Rating')

plt.ylabel('Number of ratings')

plt.suptitle('Joke #50 - Ratings/Num of ratings')
# Create a series showing the correlaton of ratings of joke#50 against the entire dataset

dataCorr = data.corrwith(data[50])
# All jokes correlation with joke#50

dataCorr.sort_values(ascending=False).head(5)
# convert the above data into a dataframe

corr_df = pd.DataFrame(data=dataCorr,columns=['Correlation'])
# drop irrelavant row

corr_df.drop('num of ratings',axis=0,inplace=True)
corr_df.sort_values('Correlation',ascending=False).head(10)
# We can see the top 10 jokes which user rated similary to joke #50. The findings indicate a medium-low correlation between them.
# Let's add count of how many times each joke has been rated
corr_df['ratings count'] = data.drop('num of ratings',axis=1).count()
# Now we can see the top most correlated jokes and how many times they were rated

corr_df.sort_values('Correlation',ascending=False).head(10)