import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
data=pd.read_csv('../input/primary_results.csv')
data.head()
aggregate=data.groupby('candidate').sum()['votes']
aggregate.name='sum_votes'
data=data.join(aggregate,on='candidate')
data.tail()
data_sort=data.sort('sum_votes',ascending=False)
data_sort.head()
data_t_c=data_sort[(data_sort.candidate=='Donald Trump') | (data_sort.candidate=='Hillary Clinton')]

data_t_c.head()
import seaborn as sns
%matplotlib inline
plt.figure(figsize=(25,20))
sns.barplot(x='state_abbreviation',y='votes',data=data_t_c,hue='candidate')

data_sort_state=data_sort[data_sort['state']=='kentucky']
sns.pairplot(data_sort,hue='party',palette="Set2", diag_kind="kde", size=4.5)
