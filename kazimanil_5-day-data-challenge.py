# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
# Dataset Import
honey = pd.read_csv("../input/honeyproduction.csv")
honey.dtypes

honey.head(6)
honey.describe().round()
plt.hist(honey['totalprod'], facecolor = 'b')
plt.xlabel("Total Production")
plt.ylabel("Frequency")
plt.title("Histogram of Honey Production in US (lbs)")
plt.grid(True)
plt.hist(honey['prodvalue'], facecolor = 'r')
plt.xlabel("Production Value")
plt.ylabel("Frequency")
plt.title("Histogram of Honey Production Value in US (lbs)")
plt.grid(True)
print("P value of an ANOVA test between 'Yield per Colony' and 'Price per Lbs' variables is {0}. Thus, they are not independent of each other."\
      .format(ttest_ind(a = honey['yieldpercol'], b = honey['priceperlb'], equal_var = False)[1]))
dims = (12, 5)
fig, ax = plt.subplots(figsize = dims)
sns.barplot(x = honey['state'], y = honey['stocks'])
honey_yearly = honey.groupby('year', as_index = False).agg({'totalprod' : 'sum'})
fig, ax = plt.subplots(figsize = dims)
sns.tsplot(data = honey_yearly['totalprod'], time = honey_yearly['year'])
honey_chisq = honey[['state', 'year']].copy()
honey_chisq['year'] = honey_chisq['year'].astype('category')
honey_chisq['state'] = honey_chisq['state'].astype('category')
honey_chisq['observed'] = honey['totalprod']
honey_chisq_pivot = honey_chisq.pivot('state', 'year').fillna(value = 0).values
print("Chi-Squared test proves that there are significant differences among states over years with a p-value of {0}.".\
      format(chi2_contingency(observed = honey_chisq_pivot)[1]))