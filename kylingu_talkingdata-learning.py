# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_gender_age_test = pd.read_csv('../input/gender_age_test.csv', dtype={'device_id': np.str})
df_gender_age_train = pd.read_csv('../input/gender_age_train.csv', dtype={'device_id': np.str})

df_app_events = pd.read_csv('../input/app_events.csv', dtype={'app_id': np.str})
df_events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str})

df_app_labels = pd.read_csv('../input/app_labels.csv', dtype={'app_id': np.str})
df_label_categories = pd.read_csv('../input/label_categories.csv')

df_phone_brands = pd.read_csv('../input/phone_brand_device_model.csv', dtype={'device_id': np.str})
df_gender_age_test.head()
df_gender_age_test.device_id.nunique(), df_gender_age_test.shape[0]
df_gender_age_train.head()
df_gender_age_train.device_id.nunique(), df_gender_age_train.shape[0]
df_gender_age_train.info()
df_gender_age_train.describe(include='all').T
df_ga_full = pd.concat([df_gender_age_train, df_gender_age_test], axis=0, sort=False)
df_ga_full.device_id.nunique()
df_events.head()
df_events.event_id.nunique(), df_events.device_id.nunique(), df_events.shape[0]
100 * (df_gender_age_test.device_id.isin(df_events.device_id.unique())).sum()/df_gender_age_test.device_id.nunique()
100 * (df_gender_age_train.device_id.isin(df_events.device_id.unique())).sum()/df_gender_age_train.device_id.nunique()
df_app_events.head()
df_app_events.event_id.nunique(), df_app_events.shape[0]
# df_gender_age_train.device_id[]
in_train_events = df_events[df_events.device_id.isin(set(df_gender_age_train.device_id) & set(df_events.device_id))]
in_train_app_events = df_app_events[df_app_events.event_id.isin(in_train_events.event_id)]
in_train_app_events.event_id.nunique(), in_train_app_events.event_id.size, len(in_train_events)
in_test_events = df_events[df_events.device_id.isin(set(df_gender_age_test.device_id) & set(df_events.device_id))]
in_test_app_events = df_app_events[df_app_events.event_id.isin(in_test_events.event_id)]
in_train_app_events.event_id.nunique(), in_train_app_events.event_id.size, len(in_train_events)
del in_train_events
del in_train_app_events
del in_test_events
del in_test_app_events
import gc
gc.collect()
df_app_labels.head()
df_app_labels.app_id.nunique(), df_app_labels.label_id.nunique(), df_app_labels.shape[0]
df_label_categories.head()
df_label_categories.category.nunique(), df_label_categories.shape[0]
df_phone_brands.head()
df_phone_brands.device_id.nunique(), df_phone_brands.shape[0]
df_phone_brands[df_phone_brands.device_id.isin(df_phone_brands.device_id.value_counts()[df_phone_brands.device_id.value_counts() > 1]\
                                               .index.tolist())].sort_values('device_id')
df_phone_brands.drop_duplicates(subset='device_id', inplace=True)
a = df_phone_brands.groupby(['device_model']).phone_brand.nunique()[df_phone_brands.groupby(['device_model']).phone_brand.nunique() > 1]
a
df_phone_brands[df_phone_brands.device_model.isin(a.index.tolist())].sort_values(['device_model', 'phone_brand'])
a.shape[0]
df_phone_brands.phone_brand = df_phone_brands.phone_brand.map(str.strip).map(str.lower)
df_phone_brands.device_model = df_phone_brands.device_model.map(str.strip).map(str.lower)
df_phone_brands.device_model = df_phone_brands.phone_brand.str.cat(df_phone_brands.device_model)
df_phone_brands.info()
df_phone_brands.describe()
df_ga_full = df_ga_full.merge(df_phone_brands, how='left', on='device_id')
df_train = df_ga_full.loc[df_ga_full.device_id.isin(df_gender_age_train.device_id.tolist())]
df_test = df_ga_full.loc[df_ga_full.device_id.isin(df_gender_age_test.device_id.tolist())]
# sns.kdeplot(df_gender_age_train.age)
fig = plt.figure(figsize=(9, 6))
sns.distplot(df_gender_age_train.age, ax=fig.gca())
plt.title('Age distribution')
sns.despine()
fig = plt.figure(figsize=(7, 4))
sns.barplot(x = df_gender_age_train.gender.value_counts().index, y=df_gender_age_train.gender.value_counts().values, ax=fig.gca())
sns.despine()
plt.title('Gender distribution')
df_gender_age_train.groupby('group').device_id.size().sort_index(ascending=False).plot.barh(title='Age Gender Group Distribution')
sns.despine()
# for brands
c = df_train.phone_brand.value_counts()
# value counts 是自动根据数量按照降序进行排序
market_share = c.cumsum()/c.sum()
# for models
c2 = df_train.device_model.value_counts()
market_share2 = c2.cumsum()/c2.sum()
ax = plt.subplot(1,2,1)
plt.gcf().set_figheight(4)
plt.gcf().set_figwidth(12)
plt.plot(market_share.values, 'b-')
plt.title('Brand share')
sns.despine()

ax = plt.subplot(1,2,2)
plt.plot(market_share2.values, 'g-')
plt.title('Model share')
sns.despine()

plt.subplots_adjust(top=0.8)
plt.suptitle('Brand and model share');
share_majority = market_share[~(market_share>0.95)].index.tolist()
share_others = market_share[market_share>0.95].index.tolist()

share_majority2 = market_share2[~(market_share2>0.60)].index.tolist()
share_others2 = market_share2[market_share2>0.60].index.tolist()
str(share_majority2)
# https://seaborn.pydata.org/tutorial/categorical.html
# sns.swarmplot(x="phone_brand", y="age", hue="gender", data=df_train);
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="phone_brand", y="age", hue="gender", data=df_train[df_train.phone_brand.isin(share_majority)].sort_values('age'), ax=fig.gca());
ax.set_xticklabels(share_majority, rotation=30);
str(share_majority)
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="device_model", y="age", hue="gender", data=df_train[df_train.device_model.isin(share_majority2)].sort_values('age'), ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);
str(share_majority2)
df_train.head()
df_app_labels.head()
# groups可以看到每个group长的样子
# df_app_labels.groupby('app_id').label_id.groups
df_app_labels = df_app_labels.groupby('app_id').label_id.apply(lambda x: ' '.join(str(s) for s in x))
df_app_labels.head()
df_app_events.head()
df_app_events ['app_lab'] = df_app_events['app_id'].map(df_app_labels)
df_app_events.head()
df_app_events = df_app_events.groupby('event_id').app_lab.apply(lambda x: ' '.join(str(s) for s in x))
df_app_events.head()
del df_label_categories
del df_app_labels
df_events.head()
df_events['app_lab'] = df_events.event_id.map(df_app_events)
df_events.head()
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
df_events['hour'] = df_events['timestamp'].dt.hour
time_large = df_events.groupby('device_id')['hour'].apply(lambda x: max(x))
time_small = df_events.groupby('device_id')['hour'].apply(lambda x: min(x))
from collections import Counter
time_most = df_events.groupby('device_id')['hour'].apply(lambda x: Counter(x).most_common(1)[0][0])
del df_app_events
df_events.app_lab = df_events.app_lab.fillna('Missing')
df_events = df_events.groupby('device_id').app_lab.apply(lambda x: ' '.join(str(s) for s in x))
df_events.head()
df_ga_full['app_lab']= df_ga_full['device_id'].map(df_events)
df_ga_full['time_most']= df_ga_full['device_id'].map(time_most)
df_ga_full['time_large']= df_ga_full['device_id'].map(time_large)
df_ga_full['time_small']= df_ga_full['device_id'].map(time_small)
df_ga_full.head()
del df_train
del df_test
del df_events
del df_phone_brands
del time_large
del time_most
del time_small
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_most", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_large", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_small", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
# 将NA当作一个类别来处理。
df_app_lab_vectorized = vectorizer.fit_transform(df_ga_full['app_lab'].fillna('Missing')) 
# 可以考虑使用label category 将feature names替换掉我们更为熟悉的文字表述。
str(vectorizer.get_feature_names())
app_labels = pd.DataFrame(df_app_lab_vectorized.toarray(), columns=vectorizer.get_feature_names(), index=df_ga_full.device_id)
app_labels.head(3)
df_ga_full = df_ga_full.merge(app_labels, how='left', left_on='device_id', right_index=True)
df_ga_full.head(3)
df_ga_full = pd.get_dummies(df_ga_full.drop(columns=['gender', 'age', 'app_lab']), columns=['phone_brand', 'device_model', 'time_most', 'time_large', 'time_small'])
df_ga_full.head(3)
df_ga_full.shape
df_ga_full.info()
df_ga_full.describe()
train = df_ga_full[df_ga_full.device_id.isin(df_gender_age_train.device_id)]
test = df_ga_full[df_ga_full.device_id.isin(df_gender_age_test.device_id)].drop(columns=['group'])

X = train.drop(columns=['group'])
encoder = LabelEncoder()
Y = encoder.fit_transform(train['group'])
X.shape, Y.shape
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
# scores = cross_val_score(LogisticRegression(), X, Y, scoring='neg_log_loss',cv=10, verbose=1)
# scores.mean(), scores
# from sklearn.cross_validation import cross_val_predict
# y_pred = cross_val_predict(LogisticRegression(), X, Y, cv=10, n_jobs=-1, verbose=1)
# log_loss(Y, y_pred)
# from sklearn.model_selection import StratifiedKFold
# kf = StratifiedKFold(n_splits=10, random_state=0)
# pred = np.zeros((Y.shape[0], Y.nunique()))
# for train_index, test_index in kf.split(X, Y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
#     lr = LogisticRegression(solver='sag').fit(X_train, y_train)
#     pred[test_index,:] = lr.predict_proba(X_test)
#     # Downsize to one fold only for kernels
#     print("{:.5f}".format(log_loss(y_test, pred[test_index, :]), end=' '))

# # log_loss(Y, pred)
import xgboost as xgb
from sklearn.model_selection import train_test_split

X.set_index('device_id', inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=.80)

##################
#     XGBoost
##################

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 12, # Y一共有12个类别
    "booster": "gbtree", # 默认为基于树的模型gbtree,还有基于线性模型的gbliner。
    "eval_metric": "mlogloss",
    "eta": 0.3, # 和GBM中的 learning rate 参数类似。
    "silent": 0, # 用于控制输出的信息，1静默模式，0默认，输出更多的，以帮助我们更好的理解。
}
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 140, evals=watchlist, verbose_eval=True)
test.set_index('device_id', inplace=True)
y_pre = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration)
# scores = cross_val_score(RandomForestClassifier(n_est
# from sklearn.ensemble import RandomForestClassifier
# scores = cross_val_score(RandomForestClassifier(n_estimators=100), X, Y, scoring='neg_log_loss',cv=10, verbose=1)
# scoresmean(), scores.
pd.read_csv('../input/sample_submission.csv').head()
result = pd.DataFrame(y_pre, index=test.index, columns=encoder.classes_)
result.head()
result.to_csv('./predict_prob.csv')
pd.read_csv('./predict_prob.csv').head()
