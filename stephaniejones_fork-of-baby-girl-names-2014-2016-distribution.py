import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df2014 = pd.read_csv('../input/girls2014.csv')
df2015 = pd.read_csv('../input/girls2015.csv')
df2016 = pd.read_csv('../input/girls2016.csv')
# so far.... imported all relevant libraries
# downloaded and imported datasets from ONS 
df2014.head()
df2015.head()
df2016.head()
df2014.info()
df2015.info()
df2016.info()
# the datasets are different sizes - in order to compare them I will keep only the top 7,437 rows of data using loc - data selection by label
df2016.loc[0:99, :]
df2014.loc[0:99, :]
df2015.loc[0:99, :]
# I want to compare the distribution of data across the 3 years, e.g. is the most popular name always similiarly distributed. 
# merge the data sets together using pd.merge
# pd.merge(df2014, df2015, df2016, how='left', on='index')

# allyears = df2014.Count, df2015.Count, df2016.Count
# allyears = pd.merge(df2014, df2015)
# allyears.head()
df2014["one"] = ""
df2014.head()
dfall=pd.merge(df2014, df2015, on = "Rank")
dfall.head()
df141516=pd.merge(dfall, df2016, on = "Rank")
df141516.head()
# dataframe, with top 100 names and their counts, ready to graph, histogram
# df141516['Count_x'].hist(bins=100)
# this plot is the wrong way round - x and y need to be swapped
# df141516.hist(column="Count_x")
# df141516.T

# df141516.hist(column="Count_y")
df141516.plot.bar(x = 'Rank', y='Count_x')
# this isnt really very useful

bplot = sns.boxplot(y='Count_x', x='Rank', 
                 data=df141516, 
                 width=0.5,
                 palette="colorblind")
# neither is this 