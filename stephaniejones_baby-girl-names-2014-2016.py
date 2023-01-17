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
df2016.loc[0:7436, :]
df2014.loc[0:7437, :]
df2015.loc[0:7436, :]
# I want to compare the distribution of data across the 3 years, e.g. is the most popular name always similiarly distributed. 
# merge the data sets together using pd.merge
# pd.merge(df2014, df2015, df2016, how='left', on='index')

