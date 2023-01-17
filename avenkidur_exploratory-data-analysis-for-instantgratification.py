import os, re, gc, warnings

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

from scipy.stats import wasserstein_distance as wd

import seaborn as sns

warnings.filterwarnings("ignore")

gc.collect()
DATA_DIR = '../input/'

FILES={}

for fn in os.listdir(DATA_DIR):

    FILES[ re.search( r'[^_\.]+', fn).group() ] = DATA_DIR + fn



CAT_COL='wheezy-copper-turtle-magic'    



train = pd.read_csv(FILES['train'],index_col='id')

test = pd.read_csv(FILES['test'],index_col='id')

CATS = sorted(train[CAT_COL].unique())



# I use id as the index column to make work easier when splitting or augmenting training data

# You can waste a lot of time with poor pandas concatenation
c=train[ train[CAT_COL]==0 ].corr().abs()

high_c = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

# unpacking that statement

# get indices for upper triangular matrix not including diagonal np.triu(np.ones(c.shape), k=1).astype(bool)

# get the indexed values, and store them in a sorted list c.where().stack().sort_values(ascending=False)

high_c
high_c.hist(bins=100)

display(high_c.describe())

print(f'''# of feature pairs with correlation greater than:\n

      0.1 : {sum(high_c.gt(.1))} \n

      0.15: {sum(high_c.gt(.15))}''')
def col_sort_by_wd(df):

    distances ={}

    for c in tqdm_notebook( range(1,df.shape[1]-1) ):

        a = df.loc[ df.target==0 ].iloc[:,c]

        b = df.loc[ df.target==1 ].iloc[:,c]

        distances[train.columns[c]]= wd(a,b)

    w = pd.Series(distances)

    return w.sort_values(ascending=False)
w1 = col_sort_by_wd(train).drop(CAT_COL)

sns.distplot(w1)

w1
w2 = col_sort_by_wd(train[ train[CAT_COL]==0 ].drop(CAT_COL,axis=1))

sns.distplot(w2)

w2
w2[ w2.gt(.25) ]

# top 45 features for separating by target, when 'wheezy-copper-turtle-magic'==0
sns.jointplot(w1,w2[w1.index],marker='1').set_axis_labels('Feature scores across whole dataset','Feature scores when restricted to wheezy-copper-turtle-magic==0')
from scipy.stats import wasserstein_distance as wd



feat='zippy-harlequin-otter-grandmaster'

sub_idx_0 = train['target']==0

sub_idx_1 = train['target']==1



sns.distplot(train[sub_idx_0][feat],bins=16,color='red')

sns.distplot(train[sub_idx_1][feat],bins=16,color='blue')



print(f'Wasserstein distance between distributions: {wd( train.loc[sub_idx_0][feat], train.loc[sub_idx_1][feat])}' )
sub_idx_0 = (train['target']==0) & (train[CAT_COL]==0)

sub_idx_1 = (train['target']==1) & (train[CAT_COL]==0)



sns.distplot(train[sub_idx_0][feat],bins=16,color='red')

sns.distplot(train[sub_idx_1][feat],bins=16,color='blue')



print(f'Wasserstein distance between distributions: {wd( train.loc[sub_idx_0][feat], train.loc[sub_idx_1][feat])}' )
c=train[ (train[CAT_COL]==0) ].corr().abs()

high_c = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

high_c.filter(like='zippy-harlequin-otter-grandmaster')

# the correlations do shift when you isolate for target==0 or 1

# ex: replace with c=train[ (train[CAT_COL]==0) & (train['target']==0) ].corr().abs()
# looking at top 2 and bottom 2 correlations with zippy-harlequin-otter-grandmaster under CAT_COL==0

cols_of_interest=['skimpy-copper-fowl-grandmaster','flaky-chocolate-beetle-grandmaster','zippy-harlequin-otter-grandmaster','blurry-wisteria-oyster-master','crabby-carmine-flounder-sorted' ]

feats = train[sub_idx_0][cols_of_interest]

sns.pairplot(train[ train[CAT_COL]==0 ],vars=cols_of_interest,markers=['1','2'],hue='target',palette='husl')

print(f'Multivariate distribution plot where {str(CAT_COL)}==0 and target==0')
names = list(train.drop(["target"],axis=1).columns.values)
first_names = []

second_names = []

third_names = []

fourth_names = []



for name in names:

    words = name.split("-")

    first_names.append(words[0])

    second_names.append(words[1])

    third_names.append(words[2])

    fourth_names.append(words[3])
fns = {x for x in first_names}

sns = {x for x in second_names}

tns = {x for x in third_names}

fons = {x for x in fourth_names}



ans = [fns, sns, tns, fons]
for i in range(len(ans)):

    for j in range(i+1,len(ans)):

        print(f'{i+1} & {j+1}: {ans[i]&ans[j]}')
display( train.filter(like='-blue-').head() )

train.filter(like='coral').head()
pd.Series(first_names).value_counts().hist()

pd.Series(first_names).value_counts()
pd.Series(second_names).value_counts().hist()

pd.Series(second_names).value_counts()
pd.Series(third_names).value_counts().hist()

pd.Series(third_names).value_counts()
pd.Series(fourth_names).value_counts().hist()

pd.Series(fourth_names).value_counts()
pd.Series(fourth_names).nunique()
train.shape
np.log(262144)/np.log(2)
len('Instant Gratification')

# Unfortunately this does not appear to be a cipher key