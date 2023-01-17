import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/kc_house_data.csv")
df.head()
df.columns
df.info()
len(df)
cor = df.corr()

sns.set(font_scale=1.25)

plt.figure(figsize=(12,12))

hm = sns.heatmap(cor, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=list(cor.columns), xticklabels=list(cor.columns))

plt.show()
plt.scatter(df.sqft_living,df.price)

plt.show()
plt.hist(df.sqft_living,bins=30)

plt.show()
plt.hist(df.price,bins=30)

plt.show()
plt.scatter(np.log(df.sqft_living),np.log(df.price))

plt.show()
(cor["price"] - cor["sqft_living"]).sort_values()
plt.hist(df.lat,bins=30)

plt.show()
cm = plt.cm.get_cmap('RdYlBu')

#cm = plt.cm.get_cmap('gnuplot')

plt.figure(figsize=(12,9))

plt.scatter(np.log(df.sqft_living),np.log(df.price),

           c = df.lat, cmap = cm, s=5)

plt.colorbar()

plt.show()
cm = plt.cm.get_cmap('RdYlBu')

#cm = plt.cm.get_cmap('gnuplot')

plt.figure(figsize=(12,9))

plt.scatter(np.log(df.sqft_living),np.log(df.price),

           c = df.lat, cmap = cm, s=5)

plt.colorbar()

plt.show()
df.waterfront.unique()
cm = plt.cm.get_cmap('RdYlBu')

#cm = plt.cm.get_cmap('gnuplot')

plt.figure(figsize=(12,9))

plt.scatter(np.log(df.sqft_living),np.log(df.price),

           c = df.lat, cmap = cm, s=(df.waterfront*40 + 5))

plt.colorbar()

plt.show()
print(df.view.unique())
#cm = plt.cm.get_cmap('Set1')

cm = plt.cm.get_cmap('RdYlBu')

#cm = plt.cm.get_cmap('gnuplot')

plt.figure(figsize=(12,9))

plt.scatter(np.log(df.sqft_living),np.log(df.price),

           c = df.lat, cmap = cm, s=(df.view+1)*20)# +1 because we don't want 0s to be invisible.

plt.colorbar()

plt.show()
f = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","yr_built","long"]
trn,tst = train_test_split(df)

lm = LinearRegression()

lm.fit(trn[f],trn["price"])

lm.score(tst[f],tst["price"])
df1 = df.copy(deep=True)

df1 = df1.drop(["date","id"],axis=1)

mapper = DataFrameMapper([(df1.columns, StandardScaler())])

scaled_features = mapper.fit_transform(df1.copy(), 4)

scaled_features_df = pd.DataFrame(scaled_features, index=df1.index, columns=df1.columns)

trn,tst=train_test_split(scaled_features_df)

lm.fit(trn[f],trn["price"])

lm.score(tst[f],tst["price"])
plt.scatter(np.log(df.price),df.condition)

plt.show()
df["family"] = (df.view+df.condition)
df.groupby(["condition","view"]).mean()["price"]/100000
plt.scatter((np.log(df.sqft_living)*(df.view+1+df.condition)),np.log(df.price),

            c=(df.view+1+df.condition),cmap = cm, s=5)

plt.show()
plt.scatter((np.log(df.sqft_living)*(df.view+df.condition)),np.log(df.price),

            c=(df.lat),cmap = cm, s=5)

plt.colorbar()

plt.show()
df.columns
f = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","yr_built","long"]
trn,tst=train_test_split(df)
lm = LinearRegression()

lm.fit(trn[f],trn["price"])

lm.score(tst[f],tst["price"])
l0 = df1.loc[df1.lat<47.45]

l1 = df1.loc[(df1.lat >= 47.45) & (df1.lat < 47.55)]

l2 = df1.loc[(df1.lat >= 47.55) & (df1.lat < 47.72)]

l3 = df1.loc[(df1.lat >= 47.72)]
print(len(l0),len(l1),len(l2),len(l3))
trn0,tst0 = train_test_split(l0)

trn1,tst1 = train_test_split(l1)

trn2,tst2 = train_test_split(l2)

trn3,tst3 = train_test_split(l3)
#x = trn0.loc[(trn0.family==7) & (trn0.lat < 45.45)]

#lm0 = RidgeCV(alphas=(0.0001,0.001,0.01,0.1,1,10))

lm0 = LinearRegression()

lm0.fit(trn0[f],trn0["price"])
lm0.score(tst0[f],tst0["price"])
lm1 = LinearRegression()

lm1.fit(trn1[f],trn1["price"])

lm1.score(tst1[f],tst1["price"])
lm2 = LinearRegression()

lm2.fit(trn2[f],trn2["price"])

lm2.score(tst2[f],tst2["price"])