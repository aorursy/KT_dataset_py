import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy.stats as stats
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
df= pd.read_csv('../input/stroopdata.csv')
df.head()
df.info()
df.describe()
df.mode()
df.median()
#Check std deviation for all population
statistics.pstdev(df['Congruent']),statistics.pstdev(df['Incongruent'])
df.mean().plot('bar');
diff_mean = df['Incongruent'].mean()-df['Congruent'].mean()
diff_mean
# Check the distribution of the congruent and incongruent groups separately

df.hist(figsize= (15,6));
# Check the distribution of the congruent and incongruent groups together

plt.figure(figsize=(12, 8))
plt.xlabel('Time spent')
plt.ylabel('Amount')
plt.hist(df['Congruent'], alpha=.7, label='Congruent')
plt.hist(df['Incongruent'], alpha=.7, label='Incongruent')
plt.title('Distribution Analysis of Congruent and Incongruent Groups')
plt.legend();
df.plot(kind='box', figsize=(12,6));
#Check the difference between all pairs and find outliers

df.plot(kind='bar', figsize=(12,6));
# Create two variables: con = congruent and inc = incongruent
con = df['Congruent'].tolist()
inc = df['Incongruent'].tolist()
# Calculate t-test

stats.ttest_rel(con,inc)
