import numpy as np 

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

from pylab import subplot

from itertools import combinations

from scipy.stats import chi2_contingency
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head(3)
num_cols = df.select_dtypes(exclude = 'O').columns

cat_cols = df.select_dtypes(include = 'O').columns

df.info()
df.describe()
plt.figure(figsize = (10, 6))

sns.distplot(df['math score'], label = 'math')

sns.distplot(df['reading score'], label = 'reading')

sns.distplot(df['writing score'], label = 'writing')

plt.xlabel('Distibutions of different scores')

plt.legend()

plt.show()
df[num_cols].corr()
plt.figure(figsize = (10, 6))

 

subplot(2, 2, 1)

sns.countplot(df['gender'])



subplot(2, 2, 2)

sns.countplot(df['lunch'])



subplot(2, 2, 3)

sns.countplot(df['race/ethnicity'])



subplot(2, 2, 4)

sns.countplot(df['test preparation course'])



plt.figure(figsize = (17, 6))

subplot(2, 2, 2)



sns.countplot(df['parental level of education'])

plt.tight_layout()    
cat_combs = combinations(cat_cols, 2)

n = len(df.index)



for pair in cat_combs:

    table_for_chi2 = pd.crosstab(df[pair[0]], df[pair[1]])

    stat, p_val, a, b = chi2_contingency(table_for_chi2)

    print('Pair: {0}\np-value of h0(independence pair attributes): {1}\n correlation = {2}\n'.format(pair, round(p_val, 5), 

                                                                                                     round(np.sqrt(stat/n), 5)))
plt.figure(figsize = (12, 5))

sns.countplot(df['race/ethnicity'], hue = df['parental level of education'])

plt.show()



plt.figure(figsize = (12, 5))

sns.countplot(df['race/ethnicity'], hue = df['gender'])

plt.show()



plt.figure(figsize = (12, 5))

sns.countplot(df['parental level of education'], hue = df['test preparation course'])

plt.show()
# binary attributes

for cat_col in ['gender', 'lunch', 'test preparation course']:

    print('\n', cat_col.upper())

    val1 = df[cat_col].unique()[0]

    val2 = df[cat_col].unique()[1]

    for col in num_cols:

        df['n'] = (df[col] - df[col].mean())/df[col].std()

        diff = df[df[cat_col] == val1]['n'].mean() - df[df[cat_col] == val2]['n'].mean()

        print('Correlation with %s =' % col, abs(round(diff, 5)))

    

del df['n']
i = 0

plt.figure(figsize = (10, 10))

for bin_col in ['gender', 'lunch', 'test preparation course']:

    for col in num_cols:

        i += 1

        subplot(3, 3, i)

        sns.violinplot(y = df[col], x = df[bin_col])

    plt.tight_layout()
# other categorical features

plt.figure(figsize = (12, 6))

sns.violinplot(x= df['parental level of education'], y = df['math score'])

plt.grid()

plt.show()



plt.figure(figsize = (12, 6))

sns.violinplot(x= df['parental level of education'], y = df['reading score'])

plt.grid()

plt.show()



plt.figure(figsize = (12, 6))

sns.violinplot(x= df['parental level of education'], y = df['writing score'])

plt.grid()

plt.show()
plt.figure(figsize = (12, 6))

sns.violinplot(x= df['race/ethnicity'], y = df['math score'])

plt.grid()

plt.show()



plt.figure(figsize = (12, 6))

sns.violinplot(x= df['race/ethnicity'], y = df['reading score'])

plt.grid()

plt.show()



plt.figure(figsize = (12, 6))

sns.violinplot(x= df['race/ethnicity'], y = df['writing score'])

plt.grid()

plt.show()
from sklearn.preprocessing import LabelEncoder



df['parental level of education'] = LabelEncoder().fit_transform(df['parental level of education'])

df['race/ethnicity'] = LabelEncoder().fit_transform(df['race/ethnicity'])



print('Correlation with parental level of education')

print(df[list(num_cols)].corrwith(df['parental level of education']))



print('\nCorrelation with race')

print(df[list(num_cols)].corrwith(df['race/ethnicity']))