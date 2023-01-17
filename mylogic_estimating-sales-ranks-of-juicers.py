import numpy as np

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import pandas as pd

import re

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from collections import Counter

import string

import operator

import seaborn as sns

from itertools import groupby

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
df_all = pd.read_csv("../input/data.csv")
df_all.index = df_all['Unnamed: 0'].values

df_all.drop('Unnamed: 0', axis=1, inplace=True)
df_all.head()
df_all.info()
(df_all=='None').sum()
df_all[df_all == 'None'] = float('NaN')
df_all.isnull().sum()
def getPrice(x):

    if type(x)==str:

        x = x.replace(',','.')

        res = re.findall('\d+.\d+', x)

        if res:

            return float(res[0])

        else:

            return float('NaN')

    else:

        return x
df_all['price'] = df_all['price'].apply(getPrice)
df_all['sales_rank'] = df_all['sales_rank'].astype(float)
df_all.dtypes
df_all['color'] = df_all['color'].str.lower()
def correct_typo(s):

    if type(s) == str:

        typo = {"sliver": "silver", "golden": "gold", "balck": "black", "sless": "stainless"}

        for k, v in typo.items():

            s = s.replace(k, v)

    return s
df_all['color'] = df_all['color'].apply(correct_typo)
def correct_brands(s):

    if type(s) == str:

        typo = {"Breville Juicer": "Breville", "Omega Juicers": "Omega"}

        for k, v in typo.items():

            s = s.replace(k, v)

    return s
df_all['brand'] = df_all['brand'].apply(correct_brands)
sns.lmplot(data=df_all, x='price', y='sales_rank')

df_all[['sales_rank', 'price']].corr()
sns.lmplot(data=df_all[df_all['price'] > 100], x='price', y='sales_rank')

df_all[df_all['price'] > 100][['sales_rank', 'price']].corr()
srank_qcut = pd.qcut(df_all['sales_rank'], 4, labels = ['0','1','2','3'])

srank_qcut.head()
srank_qcut.head()
def getColors(carr):

    clist = []

    for c in carr:

        if type(c) == str:

            lis = re.split(" |/|-|\.|,|\&", c)

            for l in lis:

                if l == '' or l == 'and' or l=='steel':

                    continue

                if l == 'sliver':

                    new_l = 'silver'

                elif l == 'golden':

                    new_l = 'gold'

                elif l == 'balck':

                    new_l = 'black'

                elif l == 'sless':

                    new_l = 'stainless'

                else:

                    new_l = l

                clist.append(new_l)

        

    return clist

serilist = []

for i in range(4):

    colors = df_all.loc[srank_qcut[srank_qcut == str(i)].index.tolist()]['color'].values

    color_list = getColors(colors)

    color_freq = dict(Counter(color_list))

    serilist.append(pd.Series(color_freq))

    #ordered_colors = sorted(color_freq.items(), key=operator.itemgetter(1), reverse=True)

df = pd.concat(serilist, axis=1)

df.dropna(inplace=True)

df.head()
top_color_list = df.index.tolist()

plt.figure(figsize=(5,5))

sns.heatmap(df, cmap='magma')
df_all.pivot_table(index='category', values='sales_rank')
plt.figure(figsize=(15,8))

sns.violinplot(x='category', y='sales_rank', data=df_all)
serilist = []

for i in range(4):

    brands = df_all.loc[srank_qcut[srank_qcut == str(i)].index.tolist()]['brand'].values.tolist()

    color_freq = dict(Counter(brands))

    serilist.append(pd.Series(color_freq))

df = pd.concat(serilist, axis=1)

plt.figure(figsize=(5,15))

sns.heatmap(df, cmap='magma')
top_brand_list = df.sort_values(by=0, ascending=False).index.tolist()[:10]

arr_brand_dummy = np.zeros(df_all.shape[0] * len(top_brand_list)).reshape(df_all.shape[0], len(top_brand_list))

df_brand_dummy = pd.DataFrame(arr_brand_dummy, columns=top_brand_list, index=df_all.index)

for row in df_all.iterrows():

    b = row[1]['brand']

    if type(b) == str:

        df_brand_dummy.loc[row[0]][row[1]['brand']] = 1

df_brand_dummy.head()
arr_color_dummy = np.zeros(df_all.shape[0] * len(top_color_list)).reshape(df_all.shape[0], len(top_color_list))

df_color_dummy = pd.DataFrame(arr_color_dummy, columns=top_color_list, index=df_all.index)

for row in df_all.iterrows():

    c = row[1]['color']

    if type(c) == str:

        colors = list(set(top_color_list).intersection(set(re.split(" |/|-|\.|,|\&", c))))

        df_color_dummy.loc[row[0]][colors] = 1

df_color_dummy.head()
cat_dummy = pd.get_dummies(df_all['category'])

cat_dummy.head()
target_sales_rank = pd.Series(np.zeros(df_all.shape[0]), index=df_all.index, name='target_sales_rank')

findex = srank_qcut[srank_qcut == '0'].index

target_sales_rank[findex] = 1
df_final = pd.concat([cat_dummy, df_brand_dummy, df_color_dummy, df_all['price'], target_sales_rank], axis=1)
df_final.dropna(inplace=True)

df_final.shape
X = df_final.drop('target_sales_rank', axis=1)

y = df_final['target_sales_rank']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lm = linear_model

models = {

    'Logistic Regression': lm.LogisticRegression, 

    'Logistic Regression CV': lm.LogisticRegressionCV, 

    'Ridge': lm.RidgeClassifier, 

    'Random Forest': RandomForestClassifier, 

    'KNN': KNeighborsClassifier

}
scores = {}

for name, model in models.items():

    clf = model()

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    scores.update({name: score})

sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

df = pd.DataFrame(sorted_scores, columns=['models', 'scores'])

df
clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)

#https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model

importance_arr = np.std(X, 0).reshape(1, X.shape[1])*clf.coef_

featurelist = X.columns.tolist()

importance = {}

for i, imp in enumerate(importance_arr[0]):

    importance.update({featurelist[i]: imp})

    

ordered_importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

df = pd.DataFrame(ordered_importance, columns=['features','importance'])

df
df['effect'] = df['importance'].apply(lambda s: 'N' if s < 0 else 'P')

df['importance'] = df['importance'].apply(np.abs)

df = df.sort_values(by='importance', ascending=False)

df