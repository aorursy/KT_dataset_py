# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



    

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/bank-additional-full.csv', sep=';')

df.head()
df[df['marital'] != 'married']['age'].mean()

df['day_of_week'].value_counts().idxmax()
df.groupby('marital')['y'].value_counts().plot(kind='bar') 


from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['default'], df['y']))



df.groupby('education')['age'].mean().plot(kind = 'bar')
from scipy.stats import pointbiserialr

pointbiserialr(df['duration'], df['age'])


df.groupby('education')['housing'].value_counts(normalize = True).plot(kind='bar') 
new_values = {'no':0,'yes':1}

df['new_ans'] = df['y'].map(new_values)

df.info()


numeric = ['age','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','euribor3m','nr.employed','new_ans']

sns.pairplot(df[numeric]);
df[numeric].corr(method='spearman')
sns.heatmap(df[numeric].corr(method='spearman'));