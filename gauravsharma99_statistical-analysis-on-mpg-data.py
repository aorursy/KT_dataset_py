# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import itertools

import pandas as pd

import scipy.stats as stats



from sklearn.preprocessing import LabelEncoder

from sklearn import feature_selection



import seaborn as sns

from matplotlib import pyplot

from statsmodels.graphics.gofplots import qqplot



sns.set()
df = pd.read_csv("../input/car-mpg/mpg_raw.csv")

df.head()
# so now the data is in rectangular form with 398 entries each having 9 distinct properties

df.shape
# let's list all the columns

columns = list(df.columns)

columns
# we now describe the properties of this dataframe like column datatype etc.

df.info()
cats = list(df.select_dtypes(include=['object']).columns)

nums = list(df.select_dtypes(exclude=['object']).columns)

print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
# let's inspect how many unique values are there in each column.

df.nunique(axis=0)
# cylinders and model_year also seems to be categorical so lets update the lists

cats.extend(['cylinders', 'model_year'])

nums.remove('cylinders')

nums.remove('model_year')



print(f'categorical variables:  {cats}')

print(f'numerical variables:  {nums}')
# check for `nans` in each column

df.isna().sum()
# let's print these 6 `nan` containing rows 

df[df.isnull().any(axis=1)]
# nan rows proportion in data

6 / len(df)
# for now remove all nan rows as they are just 1.5%

df = df[~df.isnull().any(axis=1)]

df.reset_index(inplace=True)

df.drop('index', inplace=True, axis=1)

df.shape
# find total duplicate entries and drop them if any

print(f'total duplicate rows: {df.duplicated().sum()}')



# drop duplicate rows if any

df = df[~df.duplicated()]

df.shape
# remove extra spaces if any

for col in ['origin', 'name']:

    df[col] = df[col].apply(lambda x: ' '.join(x.split()))
df['mpg_level'] = df['mpg'].apply(lambda x: 'low' if x<17 else 'high' if x>29 else 'medium')

cats.append('mpg_level')

print(f'categorical variables:  {cats}')
# before we move ahead it's a good practice to group all variables together having same type.

df = pd.concat((df[cats], df[nums]), axis=1)

df.head()
ALPHA = 0.05
# Contingency Table (aka frequency table)

pd.crosstab(df.origin, df.model_year)
observed_values = pd.crosstab(df.origin, df.mpg_level).values

observed_values
# help(stats.chi2_contingency)

chi2, p, dof, expected_values = stats.chi2_contingency(observed_values)

chi2, p, dof, expected_values
if p <= ALPHA:

    print(f'Rejected H0 under significance level {ALPHA} `origin` & `model_year` are dependent.')

else:

    print(f'Fail to reject H0 due to lack of evidence under significance level {ALPHA} `origin` & `model_year` are independent.')
# help(stats.fisher_exact)

# stats.fisher_exact doesn't support contingency table more than 2x2
df_cat_label =  pd.concat([df.loc[:, ['origin', 'mpg_level']].apply(lambda x: LabelEncoder().fit_transform(x)),

                           df.loc[: , 'cylinders': 'model_year']], axis=1)



df_cat_label.head()
chi2_res = feature_selection.chi2(df_cat_label, df.mpg_level)

df_chi2 = pd.DataFrame({

    'attr1': 'mpg_level',

    'attr2': df_cat_label.columns,

    'chi2': chi2_res[0],

    'p': chi2_res[1],

    'alpha': ALPHA

})

df_chi2['H0'] = df_chi2.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')

df_chi2['relation'] = df_chi2.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

df_chi2
nums
fig = pyplot.figure(1, (10, 4))



ax = pyplot.subplot(1,2,1)

sns.distplot(np.log2(df.mpg))

pyplot.tight_layout()



ax = pyplot.subplot(1,2,2)

sns.distplot(np.log2(df.weight))

pyplot.tight_layout()



pyplot.show()
# quantile-quantile plots on original data

fig = pyplot.figure(1, (18,8))



for i,num in enumerate(nums):

    ax = pyplot.subplot(2,3,i+1)

    qqplot(df[num], line= 's', ax=ax)

    ax.set_title(f'qqplot - {num}')

    pyplot.tight_layout()
# let's contruct a function

def shapiro_wilk_test(df: pd.DataFrame, cols: list, alpha=0.05):

    # test the null hypothesis for columns given in `cols` of the dataframe `df` under significance level `alpha`.

    for col in cols:

        _,p = stats.shapiro(df[col])

        if p <= alpha:

            print(f'''\nRejected H0 under significance level {alpha}\n{col} doesn't seems to be normally distributed''')

        else:

            print(f'''\nFail to reject H0 due to lack of evidence under significance level {alpha}\n{col} seem to be normally distributed''')
shapiro_wilk_test(df, nums)
_, p = stats.shapiro(df.acceleration)

p
from sklearn.preprocessing import PowerTransformer



df_tfnum = pd.DataFrame(PowerTransformer().fit_transform(df[nums]), columns=nums)

df_tfnum.head()
fig = pyplot.figure(1, (18,8))



for i,num in enumerate(['mpg', 'displacement', 'horsepower', 'acceleration']):

    ax = pyplot.subplot(2,3,i+1)

    sns.distplot(df_tfnum[num])

    ax.set_xlabel(f'transformed {num}')

    pyplot.tight_layout()
# quantile-quantile plots on transformed data

fig = pyplot.figure(1, (18,8))



for i,num in enumerate(['mpg', 'displacement', 'horsepower', 'acceleration']):

    ax = pyplot.subplot(2,3,i+1)

    qqplot(df_tfnum[num], line='s', ax=ax)

    ax.set_title(f'qqplot - transformed {num}')

    pyplot.tight_layout()
shapiro_wilk_test(df_tfnum, ['mpg', 'displacement', 'horsepower', 'acceleration'])
_, p = stats.shapiro(df_tfnum.acceleration)

p
for num in nums:

    if num == 'mpg':

        continue

    

    corr, p = stats.spearmanr(df.mpg, df[num])



    print(f'\n* `mpg` & `{num}`\n')

    print(f'corr: {round(corr, 4)} \t p: {p}')



    if p <= ALPHA:

        print(f'Rejected H0 under significance level {ALPHA}, mpg & {num} are correlated')

    else:

        print(f'''Fail to reject H0 due to lack of evidence under significance level {ALPHA}, 

              mpg & {num} are not correlated''')
def test_correlation(x1, x2, method='spearman', alpha=0.05):

    # this function returns correlation, p-value and H0 for `x1` & `x2`

    

    ALLOWED_METHODS = ['pearson', 'spearman', 'kendall']

    if method not in ALLOWED_METHODS:

        raise ValueError(f'allowed methods are {ALLOWED_METHODS}')

        

    if method=='pearson':

        corr, p = stats.pearsonr(x1,x2)

    elif method=='spearman':

        corr, p = stats.spearmanr(x1,x2)

    else:

        corr, p = stats.kendalltau(x1,x2)

    

    h0 = (

    'rejected'

    if p<=ALPHA else

    'fail to reject')

    

    return corr, p, h0
df_corr = pd.DataFrame(columns=['attr1', 'attr2', 'corr', 'p', 'H0'])



for combo in itertools.combinations(nums, r=2):

    corr, p, h0 = test_correlation(df[combo[0]], df[combo[1]])

    df_corr = df_corr.append({'attr1':combo[0], 'attr2':combo[1],

                              'corr':round(corr, 5), 'p':p, 'H0':h0}, ignore_index=True)

    

df_corr
shapiro_wilk_test(df[df.origin=='japan'], ['acceleration'])
shapiro_wilk_test(df[df.origin=='usa'], ['acceleration'])
# because the variance is not same for the two distributions hence equal_var=False

_, p = stats.ttest_ind(df[df.origin=='japan'].acceleration, df[df.origin=='usa'].acceleration, equal_var=False)



if p <= ALPHA:

    print(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')

else:

    print(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')
_, p = stats.f_oneway(df[df.origin=='japan'].acceleration, df[df.origin=='usa'].acceleration, df[df.origin=='europe'].acceleration)



if p <= ALPHA:

    print(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')

else:

    print(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')
shapiro_wilk_test(df[df.origin=='japan'], ['horsepower'])
shapiro_wilk_test(df[df.origin=='europe'], ['horsepower'])
shapiro_wilk_test(df[df.origin=='usa'], ['horsepower'])
_, p = stats.kruskal(df[df.origin=='japan'].horsepower, df[df.origin=='usa'].horsepower, df[df.origin=='europe'].horsepower)



if p <= ALPHA:

    print(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')

else:

    print(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')
_, p = stats.mannwhitneyu(df[df.mpg_level=='high'].acceleration, df[df.mpg_level=='medium'].acceleration)



if p <= ALPHA:

    print(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')

else:

    print(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')
acc_gb_year = df.groupby('model_year')['mpg']



acc_yr = []

for yr in df.model_year.unique():

    acc_yr.append(list(acc_gb_year.get_group(yr)))
_, p = stats.kruskal(*acc_yr)



if p <= ALPHA:

    print(f'Rejected H0 under {ALPHA*100}% significance, Different distributions.')

else:

    print(f'Fail to Reject H0 under {ALPHA*100}% significance, Same distributions.')
# help(feature_selection.f_classif)

result_f = feature_selection.f_classif(df.loc[:, 'mpg': 'acceleration'], df.cylinders)

anova_test_cat = pd.DataFrame({

    'cat-attr': 'cylinders',

    'cont-attr': df.loc[:, 'mpg': 'acceleration'].columns,

    'f': result_f[0],

    'p': result_f[1],

    'alpha': ALPHA

})

anova_test_cat['H0'] = anova_test_cat.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')

anova_test_cat['relation'] = anova_test_cat.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

anova_test_cat
result_f = feature_selection.f_classif(df_cat_label[['origin', 'cylinders', 'model_year']], df.mpg)

anova_test_cat = pd.DataFrame({

    'cont-attr': 'mpg',

    'cat-attr': ['origin', 'cylinders', 'model_year'],

    'f': result_f[0],

    'p': result_f[1],

    'alpha': ALPHA

})

anova_test_cat['H0'] = anova_test_cat.p.apply(lambda x: 'rejected' if x <= ALPHA else 'fail to reject')

anova_test_cat['relation'] = anova_test_cat.H0.apply(lambda x: 'dependent' if x=='rejected' else 'independent')

anova_test_cat