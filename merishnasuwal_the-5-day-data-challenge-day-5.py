import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import scipy.stats as sc
df = pd.read_csv('../input/cereal.csv')
df.columns
df1 = df['mfr']
df2 = df['type']
sc.chisquare(df1.value_counts())
sc.chisquare(df2.value_counts())
contingency = pd.crosstab(df1,df2)
sc.chi2_contingency(contingency)
