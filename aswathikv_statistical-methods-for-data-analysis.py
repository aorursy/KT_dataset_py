import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
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
df = pd.read_csv('/kaggle/input/statistical-learning-dataset/cs1.csv')

df.head()
male = df[df['gender'] == 'MALE']

female = df[df['gender'] == 'FEMALE']
male.income.mean()
female.income.mean()
sns.distplot(df['income'])
import scipy.stats as st



st.shapiro(df.income)
st.mannwhitneyu(male.income,female.income)
st.levene(male.income,female.income)
st.ttest_ind(male.income,female.income)
married = df[df['married'] == 'YES']

unmarried = df[df['married'] == 'NO']
st.mannwhitneyu(married.income,unmarried.income)
st.ttest_ind(married.income,unmarried.income)
ply = df[df['pl'] == 'YES']

pln = df[df['pl'] == 'NO']
st.mannwhitneyu(ply.income,pln.income)
st.ttest_ind(ply.income,pln.income)
df.region.unique()
ic = df[df['region'] == 'INNER_CITY']

t = df[df['region'] == 'TOWN']

r = df[df['region'] == 'RURAL']

s = df[df['region'] == 'SUBURBAN']
st.levene(ic.income,t.income,r.income,s.income)
sns.boxplot(df['region'],df['income'])
st.f_oneway(ic.income,t.income,r.income,s.income)
st.kruskal(ic.income,t.income,r.income,s.income)
df.children.unique()
zero = df[df['children'] == 0]

one = df[df['children'] == 1]

two = df[df['children'] == 2]

three = df[df['children'] == 3]
st.f_oneway(zero.income,one.income,two.income,three.income)
st.kruskal(zero.income,one.income,two.income,three.income)
sns.distplot(df.age)
st.shapiro(df['age'])
st.levene(male.age,female.age)
st.mannwhitneyu(male.age,female.age)
st.ttest_ind(male.age,female.age)
st.levene(married.age,unmarried.age)
st.mannwhitneyu(married.age,unmarried.age)
st.ttest_ind(married.age,unmarried.age)
st.levene(ply.age,pln.age)
st.mannwhitneyu(ply.age,pln.age)
st.ttest_ind(ply.age,pln.age)
st.levene(ic.age,t.age,r.age,s.age)
st.f_oneway(ic.age,t.age,r.age,s.age)
st.kruskal(ic.age,t.age,r.age,s.age)
st.levene(zero.age,one.age,two.age,three.age)
st.f_oneway(zero.age,one.age,two.age,three.age)
st.kruskal(zero.age,one.age,two.age,three.age)
from statsmodels.stats.proportion import proportions_ztest



ct = pd.crosstab(df['pl'],df['gender'])
ct
proportions_ztest([86,62],[170,160])
st.chi2_contingency(ct)
cm = pd.crosstab(df['pl'],df['married'])
st.chi2_contingency(cm)
c = pd.crosstab(df['pl'],df['children'])

st.chi2_contingency(c)
r = pd.crosstab(df['pl'],df['region'])

st.chi2_contingency(r)
st.chi2_contingency(pd.crosstab(df['pl'],df['car']))
st.chi2_contingency(pd.crosstab(df['pl'],df['save_act']))
st.chi2_contingency(pd.crosstab(df['pl'],df['current_act']))
import statsmodels.api as sm

from statsmodels.formula.api import ols

from statsmodels.stats.anova import anova_lm
res = ols('income ~ pl +car',data=df).fit()

res.summary()
anova_lm(res,typ=1)
res1 = ols('income ~ pl +car+pl:car',data=df).fit()

res1.summary()