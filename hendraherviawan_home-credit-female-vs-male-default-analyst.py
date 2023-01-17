# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

pd.options.display.float_format = '{:,.3f}'.format

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

i="SK_ID_CURR"
t='TARGET'
c='CODE_GENDER'
c1='AMT_CREDIT'
c2='OCCUPATION_TYPE'
c3='AMT_INCOME_TOTAL'

app = '../input/application_train.csv'

application_train = pd.read_csv(app)
#application_train["OCCUPATION_TYPE"] = application_train["OCCUPATION_TYPE"].astype('category')

application_train["log_"+c1] = np.log1p(application_train[c1])
application_train["log_"+c3] = np.log1p(application_train[c3])

application_train.info()
def percentage(part, whole):
  return 100 * float(part-whole)/float(whole)
fig, ax =plt.subplots(1,2, figsize=(20,5))
ax[0].set_title("count") #0
sns.countplot(hue=t, x=c, data=application_train, ax=ax[0])

ax[1].set_title("sum("+c1+")") #1
sns.barplot(x=c, y=c1, hue=t, estimator=np.sum, data=application_train, ax=ax[1])
#Count
group = pd.pivot_table(application_train, values=i, index=[c, t], aggfunc=[np.count_nonzero])
x = group.unstack()
x.columns = ['0','1'] # rename columns
x1 = x.loc[['F', 'M'],:].sum(axis=1).rename('total') # sum not-default+default
x1 = pd.concat([x.loc[['F', 'M'],:],x1], axis=1) # concat
x1['pct_default'] = x1['1']/x1['total'] # calculate percentage
x1
# Let's look at the density of the AMT_INCOME_TOTAL of F vs M
temp1 = application_train.loc[application_train[c] != 'XNA'] # exclude XNA

#plt.figure(figsize=(20,10))
sns.FacetGrid(temp1, hue=c, size=6) \
   .map(sns.kdeplot, "log_"+c1) \
   .add_legend()
plt.title('KDE Plot of '+c1);
#sum(AMT_CREDIT)
group = pd.pivot_table(application_train, values=c1, index=[c, t], aggfunc=[np.sum])
x = group.unstack()
x.columns = ['0','1'] # rename columns
x1 = x.loc[['F', 'M'],:].sum(axis=1).rename('total') # sum not-default+default
x1 = pd.concat([x.loc[['F', 'M'],:],x1], axis=1) # concat
x1['pct_default'] = x1['1']/x1['total'] # calculate percentage
x1
#application_train["log_"+c3] = np.log1p(application_train[c3])

fig, ax =plt.subplots(1,2, figsize=(20,5))
ax[0].set_title("Distribution of " +c3) #0
sns.distplot(application_train[c3], ax=ax[0])

ax[1].set_title("Distribution of log1p("+c3+")") #1
sns.distplot(np.log1p(application_train[c3]), ax=ax[1])
# Let's look at the density of the AMT_INCOME_TOTAL of F vs M
temp1 = application_train.loc[application_train[c] != 'XNA'] # exclude XNA

#plt.figure(figsize=(20,10))
sns.FacetGrid(temp1, hue=c, size=6) \
   .map(sns.kdeplot, "log_"+c3) \
   .add_legend()
plt.title('KDE Plot of '+c3);
pd.pivot_table(application_train, values=c3, index=[c], aggfunc=[np.count_nonzero, np.mean, np.std, np.min, np.max], margins=True)
temp1 = application_train

pd.pivot_table(temp1, values=c3, index=[c2, c], #add c to multi index by gender
               aggfunc=[np.count_nonzero, np.mean, np.std, np.min, np.max])#.sort_values(('mean', c3))
# temp1 = application_train.groupby([application_train[c2], application_train[c]]).size().reset_index().rename({0:'count'}, axis=1)
temp1 = application_train
plt.figure(figsize=(20,5))
plt.xticks(rotation=45)

sns.countplot(x=temp1.OCCUPATION_TYPE, hue=temp1.CODE_GENDER, palette="Blues_d")
temp1 = application_train #[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.boxplot(x=temp1.OCCUPATION_TYPE, y=temp1['log_AMT_INCOME_TOTAL'], hue=temp1.CODE_GENDER, palette="Blues_d")
temp1 = application_train[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.barplot(x=temp1.OCCUPATION_TYPE, y=temp1.AMT_INCOME_TOTAL, hue=temp1.CODE_GENDER, palette="Blues_d")
temp1 = application_train #[c3].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.boxplot(x=temp1.OCCUPATION_TYPE, y=temp1['AMT_CREDIT'], hue=temp1.CODE_GENDER, palette="Blues_d")
#mean of AMT_CREDIT
temp1 = application_train[c1].groupby([application_train[c2], application_train[c]]).mean().reset_index()

plt.figure(figsize=(20,5))
plt.xticks(rotation=45)
sns.barplot(x=temp1.OCCUPATION_TYPE, y=temp1.AMT_CREDIT, hue=temp1.CODE_GENDER, palette="Blues_d")
#TODO: Correlation map from above chart
#Number of borrower
t1 = application_train[c2].value_counts().rename('t1')
t2 = application_train[c2].loc[(application_train.TARGET == 0)].value_counts().rename('t2') 
t3 = application_train[c2].loc[(application_train.TARGET == 1)].value_counts().rename('t3')

temp1 = pd.concat([t3, t2, t1], axis=1)

temp1['t3_pct'] = temp1.t3/temp1.t1
#temp1['t2_pct'] = temp1.t2/temp1.t1

#del t1;del t2; del t3

temp1.sort_values('t3_pct')
#Ratio of "Total Income" : "Credit loan"
t1 = application_train[c1].groupby(application_train[c2]).mean()#.reset_index()
t2 = application_train[c3].groupby(application_train[c2]).mean()#.reset_index()
r = (t1/t2).rename('Ratio')

temp1 = pd.concat([t1,t2,r], axis=1)
temp1.sort_values('Ratio')
# gender default/total loan by gender
# * Realty agent spreed is widest 

#top X OCCUPATION_TYPE (default, F)
t1 = application_train[c2].loc[(application_train.CODE_GENDER == 'F') 
                               & (application_train.TARGET == 1)].value_counts().rename('Default')
t1b = application_train[c2].loc[(application_train.CODE_GENDER == 'F') 
                               ].value_counts().rename('Total') #& (application_train.TARGET == 0)
fdr = 'FDefaultRate'
t1_pct = (t1/t1b).rename(fdr)

#top X OCCUPATION_TYPE (default, M)
t2 = application_train[c2].loc[(application_train.CODE_GENDER == 'M') 
                               & (application_train.TARGET == 1)].value_counts().rename('Default')
t2b = application_train[c2].loc[(application_train.CODE_GENDER == 'M') 
                               ].value_counts().rename('Total') #& (application_train.TARGET == 0)
mdr  = 'MDefaultRate'
t2_pct = (t2/t2b).rename(mdr)

temp1 = pd.concat([t1,t1b,t1_pct,t2, t2b, t2_pct], axis=1)
temp1['t_delta'] = abs(temp1[fdr] - temp1[mdr])
temp1.sort_values('t_delta')
