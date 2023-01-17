# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
import scipy.stats as st

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/statistical-learning-dataset/cs1.csv")
df.head(6)
df['income'].describe()
sns.distplot(df['income'])
plt.show()
print("Sharpiro : ",st.shapiro(df.income))
df.groupby('gender')['income'].mean()
male = df[df['gender'] == 'MALE']['income']
female = df[df['gender'] == 'FEMALE']['income']
import scipy.stats as st
st.mannwhitneyu(male,female)
st.levene(male,female)
st.ttest_ind(male,female)
married = df[df['married'] == 'YES']['income']
unmarried = df[df['married'] == 'NO']['income']
df.groupby('married')['income'].mean()
st.mannwhitneyu(married,unmarried)
st.levene(married,unmarried)
st.ttest_ind(married,unmarried)
pl_yes = df[df['pl'] == 'YES']['income']
pl_no = df[df['pl'] == 'NO']['income']
df.groupby('pl')['income'].mean()
st.mannwhitneyu(pl_yes,pl_no)
st.levene(pl_yes,pl_no)
st.ttest_ind(pl_yes,pl_no)
df.region.unique()
iic = df[df['region'] == 'INNER_CITY']['income']        # iic = income of inner city
it = df[df['region'] == 'TOWN']['income']
ir = df[df['region'] == 'RURAL']['income']
isu = df[df['region'] == 'SUBURBAN']['income']
st.levene(iic,it,ir,isu)
sns.boxplot(x='region',y="income",data=df)
plt.show()
st.f_oneway(iic,it,ir,isu)
st.kruskal(iic,it,ir,isu)
df.children.unique()
c0 = df[df['children'] == 0]['income']        # c0 = 0 children
c1 = df[df['children'] == 1]['income']
c2 = df[df['children'] == 2]['income']
c3 = df[df['children'] == 3]['income']
st.levene(c0,c1,c2,c3)
sns.boxplot(x='children',y="income",data=df)
plt.show()
st.f_oneway(c0,c1,c2,c3)
st.kruskal(c0,c1,c2,c3)
from statsmodels.stats.proportion import proportions_ztest
tab = pd.crosstab(df['pl'],df['gender'])
tab = tab.T
tab
tab.YES
proportions_ztest((tab.YES[0],tab.YES[0]+tab.YES[1]),(tab.YES[1],tab.YES[0]+tab.YES[1]))
proportions_ztest([86,62],[170,160])
proportions_ztest([tab.YES[1], tab.YES[0]], [tab.YES[1]+tab.NO[1], tab.YES[0]+tab.NO[0]])
st.chi2_contingency(tab)
tab = pd.crosstab(df['pl'],df['married'])
tab = tab.T
tab
tab.YES
proportions_ztest((tab.YES[0],tab.YES[0]+tab.YES[1]),(tab.YES[1],tab.YES[0]+tab.YES[1]))
proportions_ztest([85,63],[214,116])
proportions_ztest([tab.YES[1], tab.YES[0]], [tab.YES[1]+tab.NO[1], tab.YES[0]+tab.NO[0]])
st.chi2_contingency(tab)
tab = pd.crosstab(df['pl'],df['save_act'])
tab = tab.T
tab
tab.YES
proportions_ztest((tab.YES[0],tab.YES[0]+tab.YES[1]),(tab.YES[1],tab.YES[0]+tab.YES[1]))
proportions_ztest([101,47],[103,127])
proportions_ztest([tab.YES[1], tab.YES[0]], [tab.YES[1]+tab.NO[1], tab.YES[0]+tab.NO[0]])
st.chi2_contingency(tab)
tab = pd.crosstab(df['pl'],df['current_act'])
tab = tab.T
tab
tab.YES
proportions_ztest((tab.YES[0],tab.YES[0]+tab.YES[1]),(tab.YES[1],tab.YES[0]+tab.YES[1]))
proportions_ztest([120,28],[71,259])
proportions_ztest([tab.YES[1], tab.YES[0]], [tab.YES[1]+tab.NO[1], tab.YES[0]+tab.NO[0]])
st.chi2_contingency(tab)
tab = pd.crosstab(df['pl'],df['mortgage'])
tab = tab.T
tab
tab.YES
proportions_ztest((tab.YES[0],tab.YES[0]+tab.YES[1]),(tab.YES[1],tab.YES[0]+tab.YES[1]))
proportions_ztest([46,102],[223,107])
proportions_ztest([tab.YES[1], tab.YES[0]], [tab.YES[1]+tab.NO[1], tab.YES[0]+tab.NO[0]])
st.chi2_contingency(tab)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
df1 = df.copy()
df1.head(6)
df1['pl'] = df['pl'].map({'YES':1,'NO':0})
df1['car'] = df['car'].map({'YES':1,'NO':0})
df1.head(6)
formula = 'income ~ C(pl)+ C(car)'
model = ols(formula,df1).fit()
anvo_table = anova_lm(model,typ = 2)
anvo_table
model1 = ols('income ~ pl + car',data = df1).fit()
model1.summary()
from scipy.stats import f
f.sf(17.03,2,327)