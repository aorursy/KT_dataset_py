import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn-dark')
import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter('ignore')
train = pd.read_csv("../input/machine-hack-housing-price-prediction/Train.csv")
test = pd.read_csv("../input/machine-hack-housing-price-prediction/Test.csv")
ss = pd.read_csv("../input/machine-hack-housing-price-prediction/sample_submission.csv")
ss.head(10)
train.head(5)
test.head(5)
Target_col = 'TARGET(PRICE_IN_LACS)'
print(f'\n Train contains {train.shape[0]} samples and {train.shape[1]} variables')
print(f'\n Test contains {test.shape[0]} samples and {test.shape[1]} variables')
train[Target_col].plot(kind='density', title="Density Distribution", fontsize=14, figsize=(10,6))
pd.Series(np.log1p(train[Target_col])).plot(kind='density', title='log distribution', fontsize=14, figsize=(10,4))
train[Target_col].plot(kind='box', vert=False, fontsize=14, figsize=(12,4), title='box-plot')
pd.Series(np.log1p(train[Target_col])).plot(kind='box', title='box-plot log transformation', vert=False, fontsize=14, figsize=(12,4))
train.head(1)
train.info()
percent_null_val = 100*(train.isnull().sum()/train.shape[0]).round(3)
percent_null_val.sort_values(ascending=False)
train.nunique()
train.columns
cat_col = ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 
           'BHK_OR_RK', 'READY_TO_MOVE', 'RESALE']
num_col = [col for col in train.columns if col not in cat_col]
num_col
num_col = [c for c in list(num_col) if c!='ADDRESS']
num_col
fig, axis = plt.subplots(4,2, figsize=(14,22))
axes = [ax for axes_row in axis for ax in axes_row]

for i,c in enumerate(train[cat_col]):
  _ = train[c].value_counts()[::-1].plot(kind='pie', ax=axes[i], title=c, autopct='%.0f', fontsize=12)
  _ = axes[i].set_ylabel('')

_ = plt.tight_layout()
fig, axes = plt.subplots(4,2,figsize=(16,16))
axes = [ax for axes_row in axes for ax in axes_row]

for i,c in enumerate(train[cat_col]):
  _ = train[c].value_counts()[::-1].plot(kind='barh', ax=axes[i], title=c, fontsize=12)

_ = plt.tight_layout()

sns.catplot(x='POSTED_BY', y=Target_col, data=train, height=5, aspect=24/16)
['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 
           'BHK_OR_RK', 'READY_TO_MOVE', 'RESALE']
sns.catplot(x='UNDER_CONSTRUCTION', y=Target_col, data=train, height=5, aspect=24/16)
sns.catplot(x='RERA', y=Target_col, data=train, height=5, aspect=24/16)
sns.catplot(x='BHK_OR_RK', y=Target_col, data=train, height=4, aspect=24/16)
sns.catplot(x='READY_TO_MOVE', y=Target_col, data=train, height=4, aspect=24/16)
sns.catplot(x='RESALE', y=Target_col, data=train, height=4, aspect=24/16)
fig, axes = plt.subplots(4,1,figsize=(12,18))

for i,c in enumerate(train[num_col]):
  _ = train[[c]].boxplot(ax=axes[i], vert=False)


train[num_col].head()
sns.set(font_scale=1.3)
fig, axes = plt.subplots(2,2, figsize=(18,14))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(num_col):
  train[c].plot.kde(ax=axes[i])

plt.tight_layout()
fig, axes = plt.subplots(4,1,figsize=(12,18))

for i,c in enumerate(train[num_col]):
  _ = np.log1p(train[[c]]).boxplot(ax=axes[i], vert=False)

log_data = pd.DataFrame()
for c in num_col:
    log_data[c] = np.log1p(train[c])

sns.set(font_scale=1.3)
fig, axes = plt.subplots(2,2, figsize=(18,14))
axes = [ax for axes_row in axes for ax in axes_row]

for i, c in enumerate(num_col):
    log_data[c].plot.kde(ax=axes[i])

plt.tight_layout()
plt.figure(figsize=(14,8))
_ = sns.heatmap(data=train[num_col].corr(), annot=True)
_ = sns.pairplot(train[num_col], height=5, aspect=24/16)

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(stopwords = set(list(STOPWORDS) + ['|']), random_state=42)
plt.figure(figsize=(10,6))
op = wc.generate(str(train['ADDRESS']))
plt.imshow(op)
plt.title("ADDRESS")
plt.axis('off')
train[Target_col].describe()
add_len = train['ADDRESS'].apply(lambda x: len(x))
add_len
train[Target_col].corr(add_len)
train[num_col] = train[num_col].apply(lambda x: np.log1p(x))
num_cols = [c for c in num_col if c!=Target_col]
test[num_cols] = test[num_cols].apply(lambda x: np.log1p(x))
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def rmsle(y_test, y_pred):
    return np.sqrt(mean_squared_log_error(y_test, y_pred))

def av_metric(y_true, y_pred):
    return 1000 * np.sqrt(mean_squared_error(y_true, y_pred))
  
target = train[Target_col]
pred_targ = pd.Series([target.mean()]*len(train))

av_metric_score = av_metric(target, pred_targ)

print(f'AV score is {av_metric_score}')
pred_test = pd.Series([target.mean()]*len(test))
preds_test = np.expm1(pred_test)

sol_f = pd.DataFrame()
sol_f[Target_col] = preds_test
sol_f.to_csv('sol.csv', index=False)
sol_f.head()
pred_targ = pd.Series([target.median()]*len(train))

av_metric_score = av_metric(target, pred_targ)

print(f'AV score is {av_metric_score}')
pred_test = pd.Series([target.median()]*len(test))
preds_test = np.expm1(pred_test)

sol_f = pd.DataFrame()
sol_f[Target_col] = preds_test
sol_f.to_csv('sol.csv', index=False)
sol_f.head()
num_col
target_per_squre = ((train[Target_col] + 1)/(train['SQUARE_FT'] + 1))
target_per_squre.mean()
pred_targ = train['SQUARE_FT']*0.855057

av_metric_score = av_metric(target, pred_targ)

print(f'AV score is {av_metric_score}')
pred_test = test['SQUARE_FT']*0.855057
preds_test = np.expm1(pred_test)

sol_f = pd.DataFrame()
sol_f[Target_col] = preds_test
sol_f.to_csv('sol.csv', index=False)
target_per_squre = ((train[Target_col] + 1)/(train['LATITUDE'] + 1))
target_per_squre.mean()
train['LATITUDE'] = train['LATITUDE'].fillna(train['LATITUDE'].median())
pred_targ = train['LATITUDE']*0.986532

av_metric_score = av_metric(target, pred_targ)

print(f'AV score is {av_metric_score}')
train['LATITUDE'].isnull().sum()