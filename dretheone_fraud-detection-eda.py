# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# imports

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
df = pd.read_csv("../input/diciaccio20/train1.csv")
g = sns.countplot(data=df, x='TARGET')
g.set_title('Fraud label distribution')

for patch in g.patches:
    height = patch.get_height()
    g.text( patch.get_x() + patch.get_width()/2., 
                height + 3,
               '{:1.2f}%'.format( height / len(df)*100), 
                ha="center", 
                fontsize=15) 

#checking for missing values

set(df.isnull().sum().values)
target_0 = df[df['TARGET']==0]
target_1 = df[df['TARGET']==1]
def plot_dists(col, figsize=(20, 3), **kwargs):
    f,x = plt.subplots(figsize=figsize)
    g = sns.distplot(target_0[col], **kwargs)
    g.set_title(col, fontsize=20)
    g.set(xlabel=None)

    g1 = sns.distplot(target_1[col], **kwargs)
    g1.set(xlabel=None)
    f.legend(labels=['not fraud','fraud'])
def plot_cat_dist(col, figsize=(20, 3)):
    f, x = plt.subplots(figsize=figsize)
    g = sns.countplot( x=col,  data=df)
    g.set_title(col, fontsize=20)
    g.set(xlabel=None)
    
    for patch in g.patches:
        height = patch.get_height()
        g.text(patch.get_x() + patch.get_width()/2.,
        height + 3,
        '{:1.2f}%'.format(height/df.shape[0]*100),
        ha="center",fontsize=12)
    
    gt = g.twinx()
    
    gt = sns.pointplot(x=col, y='TARGET', data=df, color='black', legend=False )
plot_dists('var15')
plot_cat_dist('var36')
plot_dists('var38')
imp_features = ['imp_aport_var13_hace3',
                 'imp_op_var39_comer_ult1',
                 'imp_op_var39_comer_ult3',
                 'imp_op_var39_efect_ult1',
                 'imp_op_var39_efect_ult3',
                 'imp_op_var39_ult1',
                 'imp_op_var41_comer_ult1',
                 'imp_op_var41_comer_ult3',
                 'imp_op_var41_efect_ult1',
                 'imp_op_var41_efect_ult3',
                 'imp_op_var41_ult1',
                 'imp_trans_var37_ult1',
                 'imp_var43_emit_ult1']
df[imp_features].describe()
f,x = plt.subplots(figsize=(20, 7))
d = df[imp_features].isin([0]).sum()/df.shape[0]
sns.barplot(x=list(d.values), y=list(d.index.values))
ind_features = ['ind_var12',
                'ind_var12_0',
                'ind_var24',
                'ind_var24_0',
                'ind_var30',
                'ind_var37',
                'ind_var37_0',
                'ind_var37_cte',
                'ind_var39_0',
                'ind_var40_0',
                'ind_var41_0',
                'ind_var5',
                'ind_var5_0',
                'ind_var8',
                'ind_var8_0']
df[ind_features].describe()
for feature in ind_features:
    plot_cat_dist(feature)

plot_cat_dist('num_med_var22_ult3')
f,x = plt.subplots(figsize=(25, 3))
g = sns.boxplot(x="num_med_var45_ult3", y="TARGET", data=df, orient='h')
g.set_title('num_med_var45_ult3', fontsize=20)
f.legend(labels=['not fraud','fraud'])
num_meses_features = ['num_meses_var12_ult3',
                     'num_meses_var39_vig_ult3',
                     'num_meses_var5_ult3',
                     'num_meses_var8_ult3']

for feature in num_meses_features:
    plot_cat_dist(feature)
df.head()
num_op_features = ['num_op_var39_comer_ult1',
         'num_op_var39_comer_ult3',
         'num_op_var39_efect_ult1',
         'num_op_var39_efect_ult3',
         'num_op_var39_hace2',
         'num_op_var39_hace3',
         'num_op_var39_ult1',
         'num_op_var39_ult3',
         'num_op_var41_comer_ult1',
         'num_op_var41_comer_ult3',
         'num_op_var41_efect_ult1',
         'num_op_var41_efect_ult3',
         'num_op_var41_hace2',
         'num_op_var41_ult1',
         'num_op_var41_ult3',
         'num_trasp_var11_ult1']

f,x = plt.subplots(figsize=(20, 7))
d = df[num_op_features].isin([0]).sum()/df.shape[0]
sns.barplot(x=list(d.values), y=list(d.index.values))
num_var_features = [ 'num_var12_0',
                     'num_var13_0',
                     'num_var13_corto_0',
                     'num_var22_hace2',
                     'num_var22_hace3',
                     'num_var22_ult1',
                     'num_var22_ult3',
                     'num_var24',
                     'num_var24_0',
                     'num_var26',
                     'num_var30',
                     'num_var30_0',
                     'num_var35',
                     'num_var37',
                     'num_var37_0',
                     'num_var37_med_ult2',
                     'num_var39_0',
                     'num_var4',
                     'num_var41_0',
                     'num_var42',
                     'num_var42_0',
                     'num_var43_emit_ult1',
                     'num_var43_recib_ult1',
                     'num_var45_hace2',
                     'num_var45_hace3',
                     'num_var45_ult1',
                     'num_var45_ult3',
                     'num_var5',
                     'num_var5_0',
                     'num_var8',
                     'num_var8_0']


f,x = plt.subplots(figsize=(20, 7))
d = df[num_var_features].isin([0]).sum()/df.shape[0]
sns.barplot(x=list(d.values), y=list(d.index.values))
df[num_var_features].describe()
for feature in num_var_features:
    plot_cat_dist(feature)
sorted(df.columns.values)
saldo_features=['saldo_medio_var12_hace2',
                 'saldo_medio_var12_ult1',
                 'saldo_medio_var12_ult3',
                 'saldo_medio_var5_hace2',
                 'saldo_medio_var5_hace3',
                 'saldo_medio_var5_ult1',
                 'saldo_medio_var5_ult3',
                 'saldo_medio_var8_ult1',
                 'saldo_medio_var8_ult3',
                 'saldo_var1',
                 'saldo_var12',
                 'saldo_var13',
                 'saldo_var13_corto',
                 'saldo_var24',
                 'saldo_var30',
                 'saldo_var37',
                 'saldo_var42',
                 'saldo_var5',
                 'saldo_var8']
for feature in saldo_features:
    plot_dists(feature, kde_kws={'bw': 0.1})
correlation = df.corr().abs()
sns.clustermap(correlation, cmap='coolwarm', 
               vmin=0, vmax=0.8, center=0, 
               square=True, linewidths=.5, 
               figsize=(50,50), yticklabels=1)
from sklearn.ensemble import RandomForestClassifier
df = df.dropna(how='any')
X = df[[col for col in df.columns.values if col!='TARGET' and col!='ID']]
y = df['TARGET']

rf = RandomForestClassifier()
rf.fit(X, y)

fv = dict(zip(X.columns, rf.feature_importances_))
fv_dict = {k: v for k, v in sorted(fv.items(), key=lambda item: item[1], reverse=True)}
fig, ax = plt.subplots(figsize=(10,20))
sns.barplot(y=list(fv_dict.keys()), x=list(fv_dict.values()))
top = list(fv_dict.keys())[:10]
top.append('TARGET')
sns.pairplot(df[top], hue='TARGET', kind="reg")