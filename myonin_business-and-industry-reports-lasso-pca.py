# Load libraries

import re

from sklearn import linear_model

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use('ggplot')
# Load data

df = pd.read_csv('../input/data.csv')
df.info()
# Drop wrong values

df.value[df.value=="S"] = None

df.value[df.value=="Z"] = None

df.value[df.value=="Less than .05 percent"] = None

df.value[df.value=="(S)"] = None

df.value[df.value=="D"] = None

df.value[df.value=="X"] = None

df.value[df.value=="A"] = None

df.date = pd.to_datetime(df['date'])

df.value = df.value.astype('float')



# Data from 2005 to 2017 

df = df.pivot_table(index='date', columns='time_series_code', values='value').reset_index()

df = df[df.date >= '2005-01-01']



# Drop colomns where NA > 2%

indx = []

for i in range(1, len(df)):

    n = len(df.iloc[:,i].isnull()[df.iloc[:,i].isnull() == True])/len(df.iloc[:,i].isnull())

    if n > 0.2:

        indx.append(i)

df = df.drop(df.columns[indx], axis=1)

df = df.dropna(1)

df.index = df.pop('date')



# Drop 'adj' columns

df = df.drop(re.findall('\w+[adj]', str(df.columns.tolist())), axis=1)
# LASSO regressor

lasso_regressor = linear_model.Lasso(max_iter=10000, random_state=1)

lasso_regressor.fit(df, df.index)

indx = [i for i, j in enumerate(lasso_regressor.coef_.tolist()) if j == 0]

df = df.drop(df.columns[indx], axis=1)
df.head()
# Normalization

df_sd = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns) 

df_sd.index = df.index.tolist()
# PCA-model

df_pca = PCA().fit(df_sd)
# Plot optimal number of PCA

plt.figure(figsize=(11, 5))

plt.plot(df_pca.explained_variance_, 'o-', markersize=12)

plt.xlabel('Numbers of PCA')

plt.ylabel('Explained variance, %')

plt.title("Optimal number of PCA -- 7")

plt.show()
# PCA dataset

df_pca = pd.DataFrame(df_pca.components_.T, columns=range(0, df_pca.n_components_), index=df_sd.columns)

df_pca = df_pca.iloc[:,0:7]
df_pca.sort_values(by = 0, ascending=False).head(4)
# Load names of rows

pca_indx = df_pca.index.tolist()

metadf = pd.read_csv('../input/metadata.csv')

metadf = metadf[metadf['time_series_code'].isin(pca_indx)].loc[:,['cat_desc', 

                                                                  'dt_desc', 

                                                                  'dt_unit', 

                                                                  'time_series_code']].drop_duplicates()

metadf.head()
# Replace codes by names

new_index = []

for i in df_pca.index:

    name_value = metadf[metadf.time_series_code==i].iloc[0,0:3].tolist()

    name_value = ', '.join(str(v) for v in name_value)

    new_index.append(name_value)

df_pca.index = new_index

new_index = []

for i in df.columns:

    name_value = metadf[metadf.time_series_code==i].iloc[0,0:3].tolist()

    name_value = ', '.join(str(v) for v in name_value)

    new_index.append(name_value)

df.columns = new_index
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 0, ascending=False).head(4).index[0]].plot(figsize=(11, 5), 

                      title = df_pca.sort_values(by = 0, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 0, ascending=False).head(4).index[1]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 0, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 0, ascending=False).head(4).index[2]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 0, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 0, ascending=False).head(4).index[3]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 0, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 1, ascending=False).head(4).index[0]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 1, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 1, ascending=False).head(4).index[1]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 1, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 1, ascending=False).head(4).index[2]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 1, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 1, ascending=False).head(4).index[3]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 1, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 2, ascending=False).head(4).index[0]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 2, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 2, ascending=False).head(4).index[1]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 2, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 2, ascending=False).head(4).index[2]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 2, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 2, ascending=False).head(4).index[3]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 2, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 3, ascending=False).head(4).index[0]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 3, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 3, ascending=False).head(4).index[1]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 3, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 3, ascending=False).head(4).index[2]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 3, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 3, ascending=False).head(4).index[3]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 3, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 4, ascending=False).head(4).index[0]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 4, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 4, ascending=False).head(4).index[1]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 4, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 4, ascending=False).head(4).index[2]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 4, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 4, ascending=False).head(4).index[3]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 4, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 5, ascending=False).head(4).index[0]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 5, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 5, ascending=False).head(4).index[1]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 5, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 5, ascending=False).head(4).index[2]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 5, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 5, ascending=False).head(4).index[3]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 5, ascending=False).head(4).index[3])

plt.show()
plt.subplot(411)

df.loc[:,df_pca.sort_values(by = 6, ascending=False).head(4).index[0]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 6, ascending=False).head(4).index[0])

plt.subplot(412)

df.loc[:,df_pca.sort_values(by = 6, ascending=False).head(4).index[1]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 6, ascending=False).head(4).index[1])

plt.subplot(413)

df.loc[:,df_pca.sort_values(by = 6, ascending=False).head(4).index[2]].iloc[:,0].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 6, ascending=False).head(4).index[2])

plt.subplot(414)

df.loc[:,df_pca.sort_values(by = 6, ascending=False).head(4).index[3]].plot(figsize=(11, 11), 

                      title = df_pca.sort_values(by = 6, ascending=False).head(4).index[3])

plt.show()