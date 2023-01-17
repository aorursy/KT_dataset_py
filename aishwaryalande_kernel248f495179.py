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
train_data = pd.read_csv("/kaggle/input/fmcgfast-moving-consumer-goods/train1.csv")
test_data = pd.read_csv("/kaggle/input/fmcgfast-moving-consumer-goods/test1.csv")
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew, kurtosis, norm

from scipy import stats

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



train_data.head(10) 

train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
print(train_data.keys())


train_data.dtypes #datatype of colunms in train data

train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)

train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)
train_data.dtypes
test_data.dtypes
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]
corr=train_data.corr()

sns.heatmap(corr,annot=True)
corr
train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['SLSMAN_CD'].count()
train_data.groupby(['PROD_CD','PLAN_MONTH'])['PROD_CD'].count()
train_data.groupby(['SLSMAN_CD','PLAN_MONTH','PLAN_YEAR'])['SLSMAN_CD'].count()
np.std(train_data) 
np.var(train_data)
skew(train_data)
kurtosis(train_data)
plt.hist(train_data['PROD_CD']);plt.title('Histogram of PROD_CD'); plt.xlabel('PROD_CD'); plt.ylabel('Frequency')
plt.hist(train_data['SLSMAN_CD'], color = 'coral');plt.title('Histogram of SLSMAN_CD'); plt.xlabel('SLSMAN_CD'); plt.ylabel('Frequency')
plt.hist(train_data['TARGET_IN_EA'], color= 'brown');plt.title('Histogram of TARGET_IN_EA'); plt.xlabel('TARGET_IN_EA'); plt.ylabel('Frequency')
plt.hist(train_data['ACH_IN_EA'], color = 'violet');plt.title('Histogram of ACH_IN_EA'); plt.xlabel('ACH_IN_EA'); plt.ylabel('Frequency')
sns.boxplot(train_data["PROD_CD"])
sns.boxplot(train_data["SLSMAN_CD"])
sns.boxplot(train_data["PLAN_MONTH"])
sns.boxplot(train_data["TARGET_IN_EA"])
sns.scatterplot(x='PROD_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PROD_CD')
sns.scatterplot(x='SLSMAN_CD', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & SLSMAN_CD')
sns.scatterplot(x='PLAN_MONTH', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_MONTH')
sns.scatterplot(x='PLAN_YEAR', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & PLAN_YEAR')
sns.scatterplot(x='TARGET_IN_EA', y='ACH_IN_EA', data=train_data).set_title('Scatterplot of ACH_IN_EA & TARGET_IN_EA')
sns.countplot(train_data["PROD_CD"])
sns.countplot(train_data["SLSMAN_CD"])

sns.countplot(train_data["PLAN_MONTH"])

sns.countplot(train_data["PLAN_YEAR"])

train_data.PROD_CD.unique()               

train_data.PROD_CD.value_counts()                    
train_data.SLSMAN_CD.unique()
train_data.SLSMAN_CD.value_counts()
train_data.PLAN_YEAR.unique()
train_data.PLAN_YEAR.value_counts()
train_data.PLAN_MONTH.unique()
train_data.PLAN_MONTH.value_counts()
train_data.TARGET_IN_EA.unique()
train_data.TARGET_IN_EA.value_counts()
train_data.ACH_IN_EA.unique()
train_data.ACH_IN_EA.value_counts()
train_data.plot(x="ACH_IN_EA",y="SLSMAN_CD")
train_data.plot(x="TARGET_IN_EA",y="SLSMAN_CD")
fig,ax= plt.subplots(figsize =(15,7))

fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['ACH_IN_EA'].unstack().plot(ax=ax)
fig,ax= plt.subplots(figsize =(15,7))

fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['ACH_IN_EA'].count().unstack().plot(ax=ax)
fig,ax= plt.subplots(figsize =(15,7))

fig= train_data.groupby(['PROD_CD','PLAN_MONTH']).count()['TARGET_IN_EA'].unstack().plot(ax=ax)
fig,ax= plt.subplots(figsize =(15,7))

fig= train_data.groupby(['SLSMAN_CD','PLAN_MONTH'])['TARGET_IN_EA'].count().unstack().plot(ax=ax)
pd.crosstab(train_data.PROD_CD,train_data.PLAN_MONTH).plot(kind="bar")

pd.crosstab(train_data.PROD_CD,train_data.PLAN_YEAR).plot(kind="bar")

pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_MONTH).plot(kind="bar")

pd.crosstab(train_data.SLSMAN_CD,train_data.PLAN_YEAR).plot(kind="bar")
sns.distplot(train_data['PROD_CD'], fit=norm, kde=False)
sns.distplot(train_data['SLSMAN_CD'], fit=norm, kde=False, color = 'coral')
sns.distplot(train_data['PLAN_MONTH'], fit=norm, kde=False, color = 'skyblue')
sns.distplot(train_data['PLAN_YEAR'], fit=norm, kde=False, color = 'orange')

sns.distplot(train_data['TARGET_IN_EA'], fit=norm, kde=False, color = 'brown')

sns.kdeplot(train_data['TARGET_IN_EA'],shade = True, bw = .5, color = "red")
import seaborn as sns
train_data["ACH_IN_EA"].describe()
sns.kdeplot(train_data['ACH_IN_EA'],shade = True, bw = .5, color = "BLUE")
sns.violinplot(y=train_data['PROD_CD'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['SLSMAN_CD'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['TARGET_IN_EA'],x=train_data['PLAN_MONTH'])
sns.violinplot(y=train_data['ACH_IN_EA'],x=train_data['PLAN_MONTH'])
target=list(train_data.TARGET_IN_EA)
achiv=list(train_data.ACH_IN_EA)
yn=[]     
for x in range(22646):

    if(target[x]<=achiv[x]):

        yn.append(1)

    else:

        yn.append(0)

train_data['result'] = yn
pd.crosstab(train_data.result,train_data.PLAN_YEAR).plot(kind="bar")
pd.crosstab(train_data.result,train_data.PLAN_MONTH).plot(kind="bar")

prod = np.array(train_data['PROD_CD'])
salesman = np.array(train_data['SLSMAN_CD'])
month = np.array(train_data['PLAN_MONTH'])
year = np.array(train_data['PLAN_YEAR'])
target = np.array(train_data['TARGET_IN_EA'])
achieved = np.array(train_data['ACH_IN_EA'])
x_ach = np.linspace(np.min(achieved), np.max(achieved))
y_ach = stats.norm.pdf(x_ach, np.mean(x_ach), np.std(x_ach))
plt.plot(x_ach, y_ach,); plt.xlim(np.min(x_ach), np.max(x_ach));plt.xlabel('achieved');plt.ylabel('Probability');plt.title('Normal Probability Distribution of achieved')
x_prod = np.linspace(np.min(prod), np.max(prod))
y_prod = stats.norm.pdf(x_prod, np.mean(x_prod), np.std(x_prod))
plt.plot(x_prod, y_prod, color = 'coral'); plt.xlim(np.min(x_prod), np.max(x_prod));plt.xlabel('prod_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of prod_cd')
x_sale = np.linspace(np.min(salesman), np.max(salesman))

y_sale = stats.norm.pdf(x_sale, np.mean(x_sale), np.std(x_sale))

plt.plot(x_sale, y_sale, color = 'coral'); plt.xlim(np.min(x_sale), np.max(x_prod));plt.xlabel('Sale_cd');plt.ylabel('Probability');plt.title('Normal Probability Distribution of sales_cd')
x_target = np.linspace(np.min(target), np.max(target))

y_target = stats.norm.pdf(x_target, np.mean(x_target), np.std(x_target))

plt.plot(x_target, y_target, color = 'coral'); plt.xlim(np.min(x_target), np.max(x_target));plt.xlabel('target');plt.ylabel('Probability');plt.title('Normal Probability Distribution of target')
train_data['PLAN_MONTH'].value_counts().head(10).plot.pie()

train_data['PLAN_YEAR'].value_counts().head(10).plot.pie()

plt.gca().set_aspect('equal')
X = train_data.iloc[:,:6]  #independent columns

y = train_data.iloc[:,-1]    #target column i.e price range
bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['imp','importance']  #naming the dataframe columns
featureScores