# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Read in both the county_facts.csv and county_facts_dictionary.csv using pd.read_csv()
county_facts = pd.read_csv("../input/county_facts.csv")
county_facts_dictionary = pd.read_csv("../input/county_facts_dictionary.csv")
# Check that these files have been read in properly by calling len() to see the number of rows.
len(county_facts), len(county_facts_dictionary)
county_facts_dictionary
# The general syntax is DataFrame[DataFrame.ColumnName == VALUE]
county_facts_dictionary[county_facts_dictionary.column_name == 'POP060210']
county_facts.head()
primary_results = pd.read_csv("../input/primary_results.csv")
primary_results.head()
Cruz = primary_results[primary_results.candidate == 'Ted Cruz']
# The .sort_values() function will put Cruz' top faction of votes at the top of the table. 
Cruz_Max = Cruz.sort_values(by= 'fraction_votes', ascending=False)
Cruz_Max.head()
Cruz_Max.tail()
# Mean proportion of the vote for Cruz, by county.
Cruz_mean = sum(Cruz_Max.fraction_votes) / len(Cruz_Max)
Cruz_mean
# Median
np.median(Cruz_Max.fraction_votes)
# So there's really not too much difference between the mean and median, though the latter is slightly greater.
# Range
Cruz_range = max(Cruz_Max.fraction_votes) - min(Cruz_Max.fraction_votes)
Cruz_range
# Standard Deviation
Cruz_std = np.std(Cruz_Max.fraction_votes)
Cruz_std
# Number of standard deviations maximum differs from the mean
(max(Cruz_Max.fraction_votes) - Cruz_mean) / Cruz_std
# Number of standard deviations minimum differs from the mean
(Cruz_mean - min(Cruz_Max.fraction_votes)) / Cruz_std
Cruz_New = pd.merge(Cruz_Max, county_facts, on='fips', how='inner')
np.corrcoef(Cruz_New.fraction_votes, Cruz_New.PST045214)
# There's a slight negative correlation, which would suggest that Cruz tends to win more of the vote in less populous counties.
# Cruz votes by population
plt.scatter(Cruz_New.PST045214, Cruz_New.fraction_votes)
plt.show()
Cruz_byPopulation = Cruz_New.sort_values(by='PST045214', ascending = False)
Cruz_PopOutlierRemoved = Cruz_byPopulation[1:172]
# Recalculate the correlation coefficient between fraction of the vote and population
np.corrcoef(Cruz_PopOutlierRemoved.fraction_votes, Cruz_PopOutlierRemoved.PST045214)
# So indeed, that outlier was subtantially diminishing the correlation!
# We can use population density as a stand-in for urban vs. rural performance. 
# POP060210 is population per square mile in 2010
np.corrcoef(Cruz_New.fraction_votes, Cruz_New.POP060210)
# There's a better correlation here.
# Cruz by population density
plt.scatter(Cruz_New.POP060210, Cruz_New.fraction_votes)
plt.show()