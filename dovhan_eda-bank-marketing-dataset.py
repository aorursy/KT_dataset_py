# The purpose of binary classification is to predict

# whether the client subscribes to a bank term deposit (variable y).



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/bank-additional-full.csv',sep=';')

df.head()
df.head(101).T

df.info()

#INFO
sns.boxplot(df['age'])#outliers - points that "get out" of the overall picture.

# Therefore it is useful to use boxplot
def outliers_indices(feature):#We will consider as outliers all points that go beyond three sigma.

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_age = outliers_indices('age')#369 outlier features will be removed from the dataset, which is not significant in this case.

out = set(wrong_age)

print(len(out))

#df.info()  # for me) alltype

#df.isnull().any() 

#print(df.columns)  #all colums

#client=df.iloc[:,0:7]

###



#print('Jobs: ', client['job'].unique())

#print('Marial: ', client['marital'].unique())



#print('Age min: ',client['age'].min())

#print('Age max: ',client['age'].max())



unmer=df.groupby('marital')['age'].mean()

print(unmer['single'],'- single ' )

print(unmer['divorced'],'- divorced ' )

df.groupby('marital')['age'].mean()

###

day=df.groupby('y')['day_of_week'].value_counts().max()

print(day, '--the table below shows that it is Monday')#always sort the table from highest to lowest





df.groupby('y')['day_of_week'].value_counts()

pd.crosstab(df['day_of_week'], df['y'])



df.groupby('y')['marital'].value_counts().plot(kind='bar') #in pairs people are smarter

#

plt.figure(figsize=(15, 5)) # change the size

sns.countplot(y='marital', hue='y', data=df); # build



#

plt.figure(figsize=(15, 5))

df.groupby('marital')['y'].value_counts().plot(kind='bar') 

plt.ylabel('y')

plt.show();

pd.crosstab(df['default'], df['poutcome'])# Consider the relationship between the deposit and the result
## identify the relationship between the availability of a loan and the result

from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['default'], df['poutcome']))

sns.heatmap(pd.crosstab(df['default'], df['poutcome']), 

            cmap="YlGnBu", annot=True, cbar=False);
plt.figure(figsize=(15, 8))

sns.countplot(y='education', hue='age', data=df);



df.groupby('education')['age'].mean().plot(kind='bar') 
from scipy.stats import pearsonr, spearmanr, kendalltau

kor = pearsonr(df['duration'], df['age'])

print('Pearson correlation:', kor[0], 'p-value:', kor[1])#That the relationship between age and is insignificant.





correlation = df.groupby('age',as_index=False)[['duration']].mean().round(2).sort_values(ascending=False,by='duration')



weight = correlation['duration']

height = correlation['age']

plt.figure(figsize=(10,8))

plt.scatter(weight,height,c='g',marker='o')

plt.xlabel('duration')

plt.ylabel('age')

plt.title('duration Vs age')

plt.show()

pd.crosstab(df['education'], df['housing'])#values at the edges are literate at the top, illiterate at the bottom

# Your code
# Default, has credit in default ?

ig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))

sns.countplot(x = 'default', data = df.iloc[: , 0:7], ax = ax1, order = ['no', 'unknown', 'yes'])

ax1.set_title('Default', fontsize=15)

ax1.set_xlabel('')

ax1.set_ylabel('Count', fontsize=15)

ax1.tick_params(labelsize=15)



# Housing, has housing loan ?

sns.countplot(x = 'housing', data = df.iloc[: , 0:7], ax = ax2, order = ['no', 'unknown', 'yes'])

ax2.set_title('Housing', fontsize=15)

ax2.set_xlabel('')

ax2.set_ylabel('Count', fontsize=15)

ax2.tick_params(labelsize=15)



# Loan, has personal loan ?

sns.countplot(x = 'loan', data = df.iloc[: , 0:7], ax = ax3, order = ['no', 'unknown', 'yes'])

ax3.set_title('Loan', fontsize=15)

ax3.set_xlabel('')

ax3.set_ylabel('Count', fontsize=15)

ax3.tick_params(labelsize=15)



plt.subplots_adjust(wspace=0.25)
print('Default:\n No credit in default:'     , df.iloc[: , 0:7][df.iloc[: , 0:7]['default'] == 'no']     ['age'].count(),

              '\n Unknown credit in default:', df.iloc[: , 0:7][df.iloc[: , 0:7]['default'] == 'unknown']['age'].count(),

              '\n Yes to credit in default:' , df.iloc[: , 0:7][df.iloc[: , 0:7]['default'] == 'yes']    ['age'].count())
print('Housing:\n No housing in loan:'     , df.iloc[: , 0:7][df.iloc[: , 0:7]['housing'] == 'no']     ['age'].count(),

              '\n Unknown housing in loan:', df.iloc[: , 0:7][df.iloc[: , 0:7]['housing'] == 'unknown']['age'].count(),

              '\n Yes to housing in loan:' , df.iloc[: , 0:7][df.iloc[: , 0:7]['housing'] == 'yes']    ['age'].count())
print('Housing:\n No to personal loan:'     , df.iloc[: , 0:7][df.iloc[: , 0:7]['loan'] == 'no']     ['age'].count(),

              '\n Unknown to personal loan:', df.iloc[: , 0:7][df.iloc[: , 0:7]['loan'] == 'unknown']['age'].count(),

              '\n Yes to personal loan:'    , df.iloc[: , 0:7][df.iloc[: , 0:7]['loan'] == 'yes']    ['age'].count())
