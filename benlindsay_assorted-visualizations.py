import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv('../input/train.csv')

train_df.head(5)
train_df.describe()
print(train_df.isnull().sum().sum())
sns.distplot(train_df['loss'], color = 'r', hist_kws={'alpha': 0.7}, kde = False)
sns.distplot(np.log(train_df['loss']), color = 'r', hist_kws={'alpha': 0.7}, kde = False)
train_df['logloss'] = np.log(train_df['loss'])

train_df.head()
#train_df_cont = train_df.filter(regex='cont.*')

train_df_cont = train_df.select_dtypes(include = ['float64'])

train_df_cont.head(5)
cols= train_df.select_dtypes(include = ['float64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(cols, vmax=1)
features = list(train_df.columns)

features.remove('id')

features.remove('loss')

cat_features = [x for x in features if x.find('cat') != -1]

cont_features = [x for x in features if x.find('cont') != -1]

correlationMatrix = train_df.copy()

correlationMatrix.drop(cat_features+['id','loss'],inplace=True,axis=1)



correlationMatrix = correlationMatrix.corr().abs()

map = sns.clustermap(correlationMatrix,annot=True,annot_kws={"size": 10})

sns.plt.setp(map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

sns.plt.show()
fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for i, ax in enumerate(axes.flatten()):

    if i==14: break

    sns.kdeplot(train_df['cont' + str(i+1)], ax=ax)
for col in train_df.columns:

    print(col + ': ' + str(len(train_df[col].unique())))
cat_dat = train_df.copy()

cat_dat.drop(cont_features,inplace=True,axis = 1)

for i in cat_features:

    try:

        sns.boxplot(data=cat_dat,x=i,y=np.log(train_df.loss))

        sns.plt.show()

    except:

        print('{} failed for some reason'.format())