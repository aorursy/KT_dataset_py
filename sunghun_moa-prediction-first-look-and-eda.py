import pandas as pd

import numpy as np

from plotnine import *

import warnings



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
train_feat = pd.read_csv('../input/lish-moa/train_features.csv')

test_feat = pd.read_csv('../input/lish-moa/test_features.csv')



train_target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_feat['type'] = 'train'

test_feat['type'] = 'test'

df = train_feat.append(test_feat)

df.shape
print(train_feat.shape)

print(test_feat.shape)

print(train_target.shape)
train_feat.head()
cols = train_feat.columns.tolist()

gtype = [col for col in cols if "g-" in col]

ctype = [col for col in cols if "c-" in col]

print("g- type(gene expression data columns count) :", len(gtype))

print("c- type(cell viabillity data columns count) :", len(ctype))
print("NA count :", train_feat.isnull().sum().sum())

print("NA count(TARGET):", train_target.isnull().sum().sum())
train_target.set_index('sig_id', inplace = True)

train_target.sum().sum() / (train_target.shape[0] * train_target.shape[1])
train_target.sum(axis = 1).value_counts()
target_count = pd.DataFrame(train_target.sum()).reset_index()

target_count.columns = ["target", "cnt"]

target_count.sort_values("cnt", ascending = False, inplace = True)

target_count.reset_index(drop = True, inplace = True)

print(target_count.head(20))



p = (ggplot(target_count[:20], aes('target', 'cnt', fill = 'target')) + geom_col(alpha = 0.5) + theme(axis_text_x=element_text(rotation=45, hjust=1)) +

     ggtitle('Top 20 Target counts'))

print(p)
def cutting_str(x):

    return x.split("_")[-1]





target_count['prop'] = target_count['cnt'] / target_count.shape[0]

target_count['target_subset'] = target_count['target'].apply(cutting_str)

target_count['target_subset'].value_counts(normalize = True).head(10)
p = (ggplot(train_feat, aes('factor(cp_type)', fill = 'factor(cp_type)')) + geom_bar(alpha = 0.5) + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cptype count')) 

print(p)

p = (ggplot(df, aes('factor(type)', fill = 'factor(cp_type)')) + geom_bar(alpha = 0.5, position="fill") + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cptype count (train vs test)')) 

print(p)
temp = train_feat.loc[train_feat.cp_type == 'ctl_vehicle']['sig_id'].values.tolist()

temp = train_target.loc[train_target.sig_id.isin(temp)]

print(temp.shape)

temp.set_index('sig_id', inplace = True)

print("ctrl_vehicle target counts :", temp.sum(axis = 1).sum())
p = (ggplot(train_feat, aes('factor(cp_time)', fill = 'factor(cp_time)')) + geom_bar(alpha = 0.5) + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cptime count')) 

print(p)

p = (ggplot(df, aes('factor(type)', fill = 'factor(cp_time)')) + geom_bar(alpha = 0.5, position="fill") + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cptime count (train vs test)')) 

print(p)
p = (ggplot(train_feat, aes('factor(cp_dose)', fill = 'factor(cp_dose)')) + geom_bar(alpha = 0.5) + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cpdose count')) 

print(p)

p = (ggplot(df, aes('factor(type)', fill = 'factor(cp_dose)')) + geom_bar(alpha = 0.5, position="fill") + theme(axis_text_x=element_text(rotation=45, hjust=1), figure_size = (5, 4)) +

     ggtitle('cpdose count (train vs test)')) 

print(p)
temp = train_feat[gtype]

temp.shape

temp = temp.iloc[:, :30].corr()

temp.reset_index(inplace= True)

temp = pd.melt(temp, id_vars= 'index')



p = ( ggplot(temp, aes(x='factor(index)', y='factor(variable)', fill='value')) + geom_tile(alpha = 1) + theme_minimal() +scale_fill_gradient2() +

     theme(figure_size = (12, 9)) + ggtitle("g-type columns corr") + theme(axis_text_x=element_text(rotation=90, hjust=1))   )



print(p)
train_feat[gtype].describe()
train_feat[ctype].describe()