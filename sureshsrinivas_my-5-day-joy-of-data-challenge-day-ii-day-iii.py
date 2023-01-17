# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import matplotlib

from matplotlib import pyplot as plt

df1 = pd.read_csv('../input/degrees-that-pay-back.csv')    #by major (50)              -- starting, median, percentile salaries

df2 = pd.read_csv('../input/salaries-by-college-type.csv') #by uni (269) / school type -- starting, median, percentile salaries

df3 = pd.read_csv('../input/salaries-by-region.csv')       #by uni (320) / region      -- starting, median, percentile salaries
df1.columns = ['major','bgn_p50','mid_p50','delta_bgn_mid','mid_p10','mid_p25','mid_p75','mid_p90']

df2.columns = ['school', 'type', 'bgn_p50','mid_p50', 'mid_p10', 'mid_p25','mid_p75','mid_p90']

df1.head()

sorted_df1 = df1.sort_values('bgn_p50', ascending=False)         

#print(sorted_df1['major'], sorted_df1['bgn_p50'])

sorted_df1.iloc[:,0:4].head(10)

def replace(df, x):

    df[x] = df[x].str.replace("$","")

    df[x] = df[x].str.replace(",","")

    df[x] = pd.to_numeric(df[x])

dollar_cols = ['bgn_p50','mid_p50','mid_p10','mid_p25','mid_p75','mid_p90']



for x in dollar_cols:

    replace(df1, x)

    replace(df2, x)



df1.describe()

df2.describe()
sorted_df2 = df2.sort_values('bgn_p50', ascending=False)         

sorted_df2.iloc[:,0:4].head(10)

import matplotlib.ticker as ticker



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter



sns.set()

majorLocator = MultipleLocator(40000)

majorFormatter = FormatStrFormatter('%d')

minorLocator = MultipleLocator(10000)

fig = plt.figure(figsize=(12, 12))

pl = ['bgn_p50', 'mid_p50', 'mid_p75', 'mid_p90']

for sp in range(0,4):

    ax = fig.add_subplot(2, 2,sp+1)

    sns.distplot(df1[pl[sp]], kde=False, rug=True)

#    sns.kdeplot(df1[pl[sp]], shade=True)

    sns.despine(top=True, right=True,left=True,bottom=True)

    ax.set_xlabel("Salary")

    ax.xaxis.set_major_locator(majorLocator)

    ax.xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter

    ax.xaxis.set_minor_locator(minorLocator)

plt.show()
fig = plt.figure(figsize=(12, 12))

pl = ['bgn_p50', 'mid_p50', 'mid_p75', 'mid_p90']

for sp in range(0,4):

    ax = fig.add_subplot(2, 2,sp+1)

    sns.set_style('white')

#    sns.distplot(df1[pl[sp]], kde=False, rug=True)

    sns.kdeplot(df1[pl[sp]], shade=True)

    sns.despine(top=True, right=True,left=True,bottom=True)

    ax.set_xlabel("Salary")

    ax.xaxis.set_major_locator(majorLocator)

    ax.xaxis.set_major_formatter(majorFormatter)

    # for the minor ticks, use no labels; default NullFormatter

    ax.xaxis.set_minor_locator(minorLocator)

plt.show()
sns.kdeplot(df1['bgn_p50'], df1['mid_p50'])
with sns.axes_style('white'):

    g = sns.jointplot("bgn_p50", "mid_p50", df2, kind='hex')

    g.ax_joint.plot(np.linspace(40000, 90000),

                    np.linspace(60000, 12000), ':k')
g1 = sns.FacetGrid(df2, col="type", size=6)

# For each subset of values, generate a kernel density plot of the "Salary" columns.

g1.map(sns.kdeplot, "bgn_p50", shade=True)



g2 = sns.FacetGrid(df2, col="type", size=6)

g2.map(sns.kdeplot, "mid_p50", shade=True)



g3 = sns.FacetGrid(df2, col="type", size=6)

g3.map(sns.kdeplot, "mid_p75", shade=True)



g4 = sns.FacetGrid(df2, col="type", size=6)

g4.map(sns.kdeplot, "mid_p90", shade=True)



plt.show()
from scipy.stats import probplot # for a qqplot

import pylab #

major_cats = ['bgn_p50', 'mid_p50', 'mid_p75', 'mid_p90']

fig = plt.figure(figsize=(10, 10))



for sp in range(0,4):

    ax = fig.add_subplot(2,2,sp+1)

    res = probplot(df2[major_cats[sp]], dist="norm", plot=ax)

    for key,spine in ax.spines.items():

        spine.set_visible(False)

    ax.set_title(major_cats[sp])

    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.show()

from scipy.stats import ttest_ind



# get the salaries for Engineering Schools

engSalaries = df2["bgn_p50"][df2["type"] == "Engineering"]

# get the salaries for Liberal Arts

libSalaries = df2["bgn_p50"][df2["type"] == "Liberal Arts"]

partySalaries = df2["bgn_p50"][df2["type"] == "Party"]



# compare them

ttest_ind(engSalaries, libSalaries, equal_var=False)

ttest_ind(engSalaries, partySalaries, equal_var=False)
# let's look at the means (averages) of each group to see which is larger

print("Mean salary for Engineering Schools:")

print(engSalaries.mean())



print("Mean salary for Liberal Arts")

print(libSalaries.mean())
# plot the Liberal Arts Salaries



plt.style.use('ggplot')

plt.hist(libSalaries, alpha=0.3, label='Liberal Arts')

# plot the engineering Salaries

plt.hist(engSalaries, alpha=0.4,  label='Engineering')



# and add a legend

plt.legend(loc='upper right')

# add a title

plt.title("Salaries by type")