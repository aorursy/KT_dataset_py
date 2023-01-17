# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
auto_df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')

auto_df.head()
auto_df.info()
auto_df = auto_df.drop(['car name'], axis = 1)
auto_df.describe()
mean = auto_df[auto_df['horsepower']!='?']['horsepower'].astype(int).mean()

print(int(mean))
def Fill_Data(data):

    if data == '?':

        return int(mean)

    else:

        return int(data)
auto_df['horsepower'] = auto_df['horsepower'].apply(Fill_Data)
auto_df.describe()
auto_df.info()
auto_df_norm = (auto_df-auto_df.min())/(auto_df.max()-auto_df.min())

auto_df_norm.head()
f, axes = plt.subplots(2, 4, figsize=(40,40))



sns.countplot(x = 'mpg', data = auto_df_norm, orient='v' ,ax = axes[0,0])

sns.countplot(x = 'cylinders', data = auto_df_norm,ax = axes[0,1])

sns.countplot(x = 'displacement', data = auto_df_norm,ax = axes[0,2])

sns.countplot(x = 'horsepower', data = auto_df_norm,ax = axes[0,3])

sns.countplot(x = 'weight', data = auto_df_norm,ax = axes[1,0])

sns.countplot(x = 'acceleration', data = auto_df_norm,ax = axes[1,1])

sns.countplot(x = 'model year', data = auto_df_norm,ax = axes[1,2])

sns.countplot(x = 'origin', data = auto_df_norm,ax = axes[1,3])



# plt.figure(figsize=(20,10))
import statsmodels.api as sm
f, axes = plt.subplots(2, 4, figsize=(40,40))



sm.qqplot(data = auto_df_norm['mpg'], fit = True, line = 's', ax = axes[0,0])

sm.qqplot(data = auto_df_norm['cylinders'], ax = axes[0,1])

sm.qqplot(data = auto_df_norm['displacement'], fit = True, line = 's', ax = axes[0,2])

sm.qqplot(data = auto_df_norm[auto_df['horsepower']!='?']['horsepower'].astype(int), fit = True, line = 's', ax = axes[0,3])

sm.qqplot(data = auto_df_norm['weight'], fit = True, line = 's', ax = axes[1,0])

sm.qqplot(data = auto_df_norm['acceleration'], fit = True, line = 's', ax = axes[1,1])

sm.qqplot(data = auto_df_norm['model year'], ax = axes[1,2])

sm.qqplot(data = auto_df_norm['origin'], ax = axes[1,3])
from scipy.stats import shapiro



def shapiro_test(df, col_list):

    for x in col_list:

        print(x)

        data = df[x]

        stat, p = shapiro(data)

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret

        alpha = 0.05

        if p > alpha:

            print('Sample looks Gaussian (fail to reject H0)')

        else:

            print('Sample does not look Gaussian (reject H0)')

        print('\n')
shapiro_test(auto_df_norm, list(auto_df_norm.columns))

# print("MPG"+"\n")

# shapiro_test(auto_df["mpg"])

# print("\n")

# print("CYLINDERS"+"\n")

# shapiro_test(auto_df["cylinders"])

# print("\n")

# print("DISPLACEMENT"+"\n")

# shapiro_test(auto_df["displacement"])

# print("\n")

# print("HORSEPOWER"+"\n")

# shapiro_test(auto_df['horsepower'])

# print("\n")

# print("WEIGHT"+"\n")

# shapiro_test(auto_df["weight"])

# print("\n")

# print("ACCELERATION"+"\n")

# shapiro_test(auto_df["acceleration"])

# print("\n")

# print("MODEL YEAR"+"\n")

# shapiro_test(auto_df["model year"])

# print("\n")

# print("ORIGIN"+"\n")

# shapiro_test(auto_df["origin"])

# print("\n")
from scipy.stats import normaltest



def D_A_test(df, col_list):

    for x in col_list:

        print(x)

        data = df[x]

        stat, p = normaltest(data)

        print('Statistics=%.3f, p=%.3f' % (stat, p))

        # interpret

        alpha = 0.05

        if p > alpha:

            print('Sample looks Gaussian (fail to reject H0)')

        else:

            print('Sample does not look Gaussian (reject H0)')

        print('\n')
D_A_test(auto_df_norm, list(auto_df_norm.columns))

# print("MPG"+"\n")

# D_A_test(auto_df["mpg"])

# print("\n")

# print("CYLINDERS"+"\n")

# D_A_test(auto_df["cylinders"])

# print("\n")

# print("DISPLACEMENT"+"\n")

# D_A_test(auto_df["displacement"])

# print("\n")

# print("HORSEPOWER"+"\n")

# D_A_test(auto_df['horsepower'])

# print("\n")

# print("WEIGHT"+"\n")

# D_A_test(auto_df["weight"])

# print("\n")

# print("ACCELERATION"+"\n")

# D_A_test(auto_df["acceleration"])

# print("\n")

# print("MODEL YEAR"+"\n")

# D_A_test(auto_df["model year"])

# print("\n")

# print("ORIGIN"+"\n")

# D_A_test(auto_df["origin"])

# print("\n")
from scipy.stats import anderson



def A_D_test(df, col_list):

    for x in col_list:

        print(x)

        data = df[x]

        result = anderson(data)

        print('Statistic: %.3f' % result.statistic)

        p = 0

        for i in range(len(result.critical_values)):

            sl, cv = result.significance_level[i], result.critical_values[i]

            if result.statistic < result.critical_values[i]:

                print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

            else:

                print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))

        print('\n')
A_D_test(auto_df_norm, list(auto_df_norm.columns))

# print("MPG"+"\n")

# A_D_test(auto_df["mpg"])

# print("\n")

# print("CYLINDERS"+"\n")

# A_D_test(auto_df["cylinders"])

# print("\n")

# print("DISPLACEMENT"+"\n")

# A_D_test(auto_df["displacement"])

# print("\n")

# print("HORSEPOWER"+"\n")

# A_D_test(auto_df['horsepower'])

# print("\n")

# print("WEIGHT"+"\n")

# A_D_test(auto_df["weight"])

# print("\n")

# print("ACCELERATION"+"\n")

# A_D_test(auto_df["acceleration"])

# print("\n")

# print("MODEL YEAR"+"\n")

# A_D_test(auto_df["model year"])

# print("\n")

# print("ORIGIN"+"\n")

# A_D_test(auto_df["origin"])

# print("\n")
auto_df_new = auto_df

auto_df.head()
auto_df_new['mpg'], bins = pd.cut(auto_df['mpg'], bins = 35, labels = False, retbins = True)

auto_df_new.head()
plt.figure(figsize = (10,10))

sns.countplot(x = 'mpg', data = auto_df_new, orient='v')
D_A_test(auto_df_new, ['mpg'])
auto_df_new['acceleration'], bins = pd.cut(auto_df['acceleration'], bins = 18, labels = False, retbins = True)



plt.figure(figsize = (10,10))

sns.countplot(x = 'acceleration', data = auto_df_new, orient='v')
sm.qqplot(data = auto_df_new['acceleration'], fit = True, line = 's')
A_D_test(auto_df_new, ['acceleration'])
from sklearn.model_selection import train_test_split



train, test = train_test_split(auto_df_norm, test_size = 0.3, random_state = 10)
train
test
print("TRAIN SET STATISTICS")

print("\n")

print(train.describe())

print("\n")

print("TEST SET STATISTICS")

print("\n")

print(test.describe())
from scipy.stats import skew



def skew_check(df, df_train, df_test, col_list):

    for x in col_list:

        print(x)

        data0 = df[x]

        data1 = df_train[x]

        data2 = df_test[x]

        skew0 = skew(data0)

        skew1 = skew(data1)

        skew2 = skew(data2)

        print('orig = %.3f,train = %.3f, test = %.3f' % (skew0, skew1, skew2))

        print('\n')
skew_check(auto_df_norm, train, test, list(auto_df_norm.columns))
from scipy.stats import kurtosis



def kurt_check(df, df_train, df_test, col_list):

    for x in col_list:

        print(x)

        data0 = df[x]

        data1 = df_train[x]

        data2 = df_test[x]

        kurt0 = kurtosis(data0)

        kurt1 = kurtosis(data1)

        kurt2 = kurtosis(data2)

        print('orig = %.3f, train = %.3f, test=%.3f' % (kurt0, kurt1, kurt2))

        print('\n')
kurt_check(auto_df_norm, train, test, list(auto_df_norm.columns))
plt.figure(figsize = (12,10))

sns.heatmap(auto_df_norm.corr(), annot = True)
from scipy.stats import chi2_contingency 



def chi2_test(table):

    stat, p, dof, expected = chi2_contingency(table)

    print('stat=%.3f, p=%.3f' % (stat, p))

    if p > 0.05:

        print('Probably independent')

    else:

        print('Probably dependent')
columns_list = list(auto_df_new.columns)
#columns_list.remove('car name')

#print(columns_list)

cont_columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
from scipy.stats import pearsonr

from itertools import combinations



def pearson_test(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = pearsonr(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.1:

            print('Probably independent')

            print('\n')

        else:

            print('Probably dependent')

            print('\n')
pearson_test(auto_df, list(auto_df.columns))
pearson_test(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import ttest_ind



def t_test(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = ttest_ind(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

            print('\n')

        else:

            print('Probably different distributions')

            print('\n')

t_test(auto_df, list(auto_df.columns))
t_test(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import f_oneway



def anova(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = f_oneway(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

        else:

            print('Probably different distributions')

        print('\n')

            
anova(auto_df, list(auto_df.columns))
anova(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import mannwhitneyu



def mannwu_test(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = mannwhitneyu(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

        else:

            print('Probably different distributions')

        print('\n')
mannwu_test(auto_df, list(auto_df.columns))
mannwu_test(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import wilcoxon



def wilc_test(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = wilcoxon(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

        else:

            print('Probably different distributions')

        print('\n')
wilc_test(auto_df, list(auto_df.columns))
wilc_test(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import kruskal



def kruskal_test(df, col_list):

    dummy = list(combinations(col_list, 2))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #rint(data2)

        stat, p = kruskal(data1, data2)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

        else:

            print('Probably different distributions')

        print('\n')
kruskal_test(auto_df, list(auto_df.columns))
kruskal_test(auto_df_norm, list(auto_df_norm.columns))
from scipy.stats import friedmanchisquare



def fcs_test(df, col_list):

    dummy = list(combinations(col_list, 3))

    dummy = list((list(x) for x in dummy))

    for x in dummy:

        print(x)

        data1 = df[x[0]].values

        #print(data1)

        data2 = df[x[1]].values

        #print(data2)

        data3 = df[x[2]].values

        stat, p = friedmanchisquare(data1, data2, data3)

        print('stat=%.3f, p=%.3f' % (stat, p))

        if p > 0.05:

            print('Probably the same distribution')

        else:

            print('Probably different distributions')

        print('\n')
fcs_test(auto_df, list(auto_df.columns))
fcs_test(auto_df_norm, list(auto_df_norm.columns))