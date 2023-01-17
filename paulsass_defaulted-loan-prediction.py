import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns



from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve

from catboost import Pool, CatBoostClassifier

from scipy.stats import pearsonr, chi2_contingency

from itertools import combinations

from statsmodels.stats.proportion import proportion_confint



%matplotlib inline
data = pd.read_csv('../input/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv',

    parse_dates=['issue_d'], infer_datetime_format=True)

data = data[(data.issue_d >= '2017-01-01 00:00:00') & (data.issue_d < '2019-01-01 00:00:00')]

data = data.reset_index(drop=True)

data.head()

#data=data.sample(1000)
data.describe()
data.columns.values
data.shape
X = data.copy()

X.info()
X.select_dtypes('object').head()
X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'], infer_datetime_format=True)

X['sec_app_earliest_cr_line'] = pd.to_datetime(X['sec_app_earliest_cr_line'], infer_datetime_format=True)
X['emp_length'] = X['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})

X['emp_length'] = X['emp_length'].str.extract('(\d+)').astype('float')

X['id'] = X['id'].astype('float')
nan_mean = X.isna().mean()

nan_mean = nan_mean[nan_mean != 0].sort_values()

nan_mean
X = X.drop(['desc', 'member_id'], axis=1, errors='ignore')
fill_empty = ['emp_title', 'verification_status_joint']

fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',

            'mths_since_last_major_derog', 'mths_since_last_record',

            'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',

            'mths_since_recent_inq', 'mths_since_recent_revol_delinq',

            'pct_tl_nvr_dlq','sec_app_mths_since_last_major_derog']

fill_min = np.setdiff1d(X.columns.values, np.append(fill_empty, fill_max))



X[fill_empty] = X[fill_empty].fillna('')

X[fill_max] = X[fill_max].fillna(X[fill_max].max())

X[fill_min] = X[fill_min].fillna(X[fill_min].min())
num_feat = X.select_dtypes('number').columns.values

X[num_feat].nunique().sort_values()
X = X.drop(['num_tl_120dpd_2m', 'id'], axis=1, errors='ignore')
num_feat = X.select_dtypes('number').columns.values

comb_num_feat = np.array(list(combinations(num_feat, 2)))

corr_num_feat = np.array([])

for comb in comb_num_feat:

    corr = pearsonr(X[comb[0]], X[comb[1]])[0]

    corr_num_feat = np.append(corr_num_feat, corr)
high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.9]

high_corr_num
X = X.drop(np.unique(high_corr_num[:, 1]), axis=1, errors='ignore')
cat_feat = X.select_dtypes('object').columns.values

X[cat_feat].nunique().sort_values()
X = X.drop(['url', 'emp_title'], axis=1, errors='ignore')
cat_feat = X.select_dtypes('object').columns.values

comb_cat_feat = np.array(list(combinations(cat_feat, 2)))

corr_cat_feat = np.array([])

for comb in comb_cat_feat:

    table = pd.pivot_table(X, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)

    corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )

    corr_cat_feat = np.append(corr_cat_feat, corr)
high_corr_cat = comb_cat_feat[corr_cat_feat >= 0.9]

high_corr_cat
X = X.drop(np.unique(high_corr_cat[:, 0]), axis=1, errors='ignore')
X.shape
keep_list = ['addr_state', 'annual_inc', 'dti', 'earliest_cr_line', 'emp_length', 'fico_range_low', 'home_ownership', 'initial_list_status', 'int_rate', 'loan_amnt', 'loan_status','mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_inq', 'mort_acc', 'pub_rec', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status']

print(keep_list)

print(len(keep_list))
drop_list = [col for col in X.columns if col not in keep_list]

print(drop_list)

print(len(drop_list))
w = [col for col in keep_list if col not in X.columns]

print(w)
X.drop(labels=drop_list, axis=1, inplace=True)
X.columns
X.shape
X.head()
X.describe()
#Cut down annual income

X = X[X['annual_inc'] < 350000]

#Get rid of missing dti i.e. 999

X = X[X['dti'] < 100]

#Mortgage accounts <= 10

X = X[X['mort_acc'] <= 10]

#Delete few entries with large amount of public deragatory marks

X = X[X['pub_rec'] <= 6]

#Trim entries with revolving utilization greater than 100%

X = X[X['revol_util'] <= 100]

#Trim entries with revolving balance greater than 200K

X = X[X['revol_bal'] <= 200000]
X['loan_status'].value_counts()
X.loc[X['loan_status'] == 'Current', 'loan_status'] = 1

X.loc[X['loan_status'] == 'Fully Paid', 'loan_status'] = 1

X.loc[X['loan_status'] == 'In Grace Period', 'loan_status'] = 1

X.loc[X['loan_status'] == 'Charged Off', 'loan_status'] = 0

X.loc[X['loan_status'] == 'Late (31-120 days)', 'loan_status'] = 0

X.loc[X['loan_status'] == 'Late (16-30 days)', 'loan_status'] = 0

X.loc[X['loan_status'] == 'Default', 'loan_status'] = 0
X['loan_status'] = X['loan_status'].astype(int)
X['loan_status'].value_counts()
XX = X.sample(10000)
g = sns.PairGrid(XX.select_dtypes('number'), hue="loan_status", hue_kws={"alpha": [0.01,0.01]}, diag_sharey=False)

g = g.map_diag(sns.kdeplot, bw=2, shade=True)

g = g.map_lower(plt.scatter)

g = g.add_legend()
XXX = X.copy()

kjk = XXX.pop('loan_status')

XXX = XXX.select_dtypes(exclude=['object', 'datetime'])

XXX.insert(XXX.shape[1], 'loan_status', kjk)
#fig, axes = plt.subplots(nrows=14, ncols=14, figsize=(25,25))

#for i in range(0,14):

#    for j in range(0,14):

#        if XXX.dtypes[j] == np.float64:

#            sns.kdeplot(XXX[XXX['loan_status'] == 1][XXX.columns[i]], XXX[XXX['loan_status'] == 1][XXX.columns[j]], cmap="Blues", shade=True, shade_lowest=False, alpha = .5, ax=axes[i][j])

#            sns.kdeplot(XXX[XXX['loan_status'] == 0][XXX.columns[i]], XXX[XXX['loan_status'] == 0][XXX.columns[j]], cmap="Reds", shade=True, shade_lowest=False, alpha = .5, ax=axes[i][j])



#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

#sns.kdeplot(XXX[XXX['loan_status'] == 'Good']['dti'], XXX[XXX['loan_status'] == 'Good']['fico_range_low'], cmap="Blues", shade=True, shade_lowest=False, alpha = .5, ax=ax)

#sns.kdeplot(XXX[XXX['loan_status'] == 'Bad']['dti'], XXX[XXX['loan_status'] == 'Bad']['fico_range_low'], cmap="Reds", shade=True, shade_lowest=False, alpha = .5, ax=ax)

#plt.show()
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20,20))

for i, ax in enumerate(axes.flatten()):

    if XXX.dtypes[i] == np.float64:

        sns.violinplot(x='loan_status', y=XXX.columns[i], data=XXX, ax=ax)
XXXX = X.copy()



X_num = XXXX.select_dtypes(exclude=['object', 'datetime'])

X_cat = XXXX.select_dtypes(exclude=['float64'])



for i in range(0, X_cat.shape[1]):

    X_num[X_cat.columns[i]] = X_cat[X_cat.columns[i]].values
#fig, axes = plt.subplots(nrows=10, ncols=14, figsize=(30,30))

#for i in range(14,24):

#    for j in range(0,14): 

#        sns.violinplot(x=X_num.columns[i], y=X_num.columns[j], data=X_num, ax=axes[i-14][j])
#fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(15,50))

#for i in range(14,23):

#    g = sns.countplot(x=X_num.columns[i], data=X_num[X_num['loan_status'] == 1], ax=axes[i-14][0], alpha = .5)

#    g = sns.countplot(x=X_num.columns[i], data=X_num[X_num['loan_status'] == 0], ax=axes[i-14][1], alpha = .5)

#    plt.setp(g.get_xticklabels(), rotation=45)

    
X.hist(figsize=(24,24))
corr = X.corr()

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
data['loan_status'].value_counts()
y = data['loan_status'].copy()

y = y.isin(['Current', 'Fully Paid', 'In Grace Period']).astype('int')

y.value_counts()
cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(data['loan_status'], data['grade']).style.background_gradient(cmap = cm)
#X_mod = X[X.grade == 'G'].copy()

X_mod = X.copy()

X_mod = X_mod.drop('loan_status', axis=1, errors='ignore')

X_mod = X_mod.drop(['sub_grade', 'int_rate'], axis=1, errors='ignore')

X_mod = X_mod.drop(['earliest_cr_line'], axis=1, errors='ignore')

y_mod = y[X_mod.index]



X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, stratify=y_mod, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=0)
X_train.head(50)
X_train.dtypes == 'object'
(((data.loc[(data['loan_status'] != 'Current') & (data.issue_d >= '2018-01-01 00:00:00'), 'total_pymnt'].sum()

   -data.loc[(data['loan_status'] != 'Current') & (data.issue_d >= '2018-01-01 00:00:00'), 'funded_amnt'].sum())

  /(data.loc[(data['loan_status'] != 'Current') & (data.issue_d >= '2018-01-01 00:00:00'), 'funded_amnt'].sum())))*100
#cat_feat_ind = (X_train.dtypes == 'object').nonzero()[0]

cat_feat_ind = [1,3,5,6,7,14]

pool_train = Pool(X_train, y_train, cat_features=cat_feat_ind)

pool_val = Pool(X_val, y_val, cat_features=cat_feat_ind)

pool_test = Pool(X_test, y_test, cat_features=cat_feat_ind)



n = y_train.value_counts()

model = CatBoostClassifier(learning_rate=.5,

                           iterations=350,

                           depth=3,

                           l2_leaf_reg=1,

                           random_strength=1,

                           bagging_temperature=1,

                           #grow_policy='Lossguide',

                           #min_data_in_leaf=1,

                           #max_leaves=1,

                           early_stopping_rounds=50,

                           class_weights=[1, n[0] / n[1]],

                           verbose=False,

                           random_state=0)

model.fit(pool_train, eval_set=pool_val, plot=True);
y_pred_test = model.predict(pool_test)



acc_test = accuracy_score(y_test, y_pred_test)

prec_test = precision_score(y_test, y_pred_test)

rec_test = recall_score(y_test, y_pred_test)

print(f'''Accuracy (test): {acc_test:.3f}

Precision (test): {prec_test:.3f}

Recall (test): {rec_test:.3f}''')



cm = confusion_matrix(y_test, y_pred_test)

ax = sns.heatmap(cm, cmap='viridis_r', annot=True, fmt='d', square=True)

ax.set_xlabel('Predicted')

ax.set_ylabel('True');
feat = model.feature_names_

imp = model.feature_importances_

df = pd.DataFrame({'Feature': feat, 'Importance': imp})

df = df.sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=df);
corr = X_mod[df['Feature'].values].corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10,10))

sns.heatmap(corr, mask=mask, square=True, cmap='RdBu_r', vmin=-1, vmax=1, annot=True, fmt='.2f');
#for calculating correct number of bins

Q1 = X_mod['fico_range_low'].quantile(0.25)

Q3 = X_mod['fico_range_low'].quantile(0.75)

IQR = Q3 - Q1

h=2*IQR*X_mod.shape[0]**(-1/3)

bins = print(max(X_mod['fico_range_low'])-min(X_mod['fico_range_low'])/h)

good = X_mod.loc[y_mod == 1, 'fico_range_low']

bad = X_mod.loc[y_mod == 0, 'fico_range_low']

sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)

ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)

ax.set_ylabel('Density')

ax.legend();
#for calculating correct number of bins

Q1 = X_mod['loan_amnt'].quantile(0.25)

Q3 = X_mod['loan_amnt'].quantile(0.75)

IQR = Q3 - Q1

h=2*IQR*X_mod.shape[0]**(-1/3)

bins = print(max(X_mod['loan_amnt'])-min(X_mod['loan_amnt'])/h)



good = X_mod.loc[y_mod == 1, 'loan_amnt']

bad = X_mod.loc[y_mod == 0, 'loan_amnt']

sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)

ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)

ax.set_ylabel('Density')

ax.legend();
#for calculating correct number of bins

Q1 = X_mod['mths_since_recent_inq'].quantile(0.25)

Q3 = X_mod['mths_since_recent_inq'].quantile(0.75)

IQR = Q3 - Q1

h=2*IQR*X_mod.shape[0]**(-1/3)

bins = print(max(X_mod['mths_since_recent_inq'])-min(X_mod['mths_since_recent_inq'])/h)



good = X_mod.loc[y_mod == 1, 'mths_since_recent_inq']

bad = X_mod.loc[y_mod == 0, 'mths_since_recent_inq']

sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)

ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)

ax.set_ylabel('Density')

ax.legend();
#for calculating correct number of bins

Q1 = X_mod['annual_inc'].quantile(0.25)

Q3 = X_mod['annual_inc'].quantile(0.75)

IQR = Q3 - Q1

h=2*IQR*X_mod.shape[0]**(-1/3)

bins = print(max(X_mod['annual_inc'])-min(X_mod['annual_inc'])/h)



good = X_mod.loc[y_mod == 1, 'annual_inc']

bad = X_mod.loc[y_mod == 0, 'annual_inc']

sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)

ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)

ax.set_ylabel('Density')

ax.legend();
y_proba_val = model.predict_proba(pool_val)[:, 1]

p_val, r_val, t_val = precision_recall_curve(y_val, y_proba_val)

plt.plot(r_val, p_val)

plt.xlabel('Recall')

plt.ylabel('Precision');
y_proba_val = model.predict_proba(pool_val)[:, 1]

fpr, tpr, thresh = roc_curve(y_val, y_proba_val)

plt.plot(fpr, tpr)

plt.xlabel('Recall')

plt.ylabel('Precision');
p_max = p_val[p_val != 1].max()

r_max = r_val[r_val != 0].max()

t_all = np.insert(t_val, 0, 0)

#t_adj_val = t_all[p_val == p_max]

t_adj_val = t_all[r_val == r_max]

y_adj_val = (y_proba_val > t_adj_val).astype(int)

p_adj_val = precision_score(y_val, y_adj_val)

print(f'Adjusted precision (validation): {p_adj_val:.3f}')
n = y_adj_val.sum()

ci = proportion_confint(p_adj_val * n, n, alpha=0.05, method='wilson')

print(f'95% confidence interval for adjusted precision: [{ci[0]:.3f}, {ci[1]:.3f}]')
y_proba_test = model.predict_proba(pool_test)[:, 1]

y_adj_test = (y_proba_test > t_adj_val).astype(int)

p_adj_test = precision_score(y_test, y_adj_test)

r_adj_test = recall_score(y_test, y_adj_test)

print(f'''Adjusted precision (test): {p_adj_test:.3f}

Adjusted recall (test): {r_adj_test:.3f}''')



cm_test = confusion_matrix(y_test, y_adj_test)

ax = sns.heatmap(cm_test, cmap='viridis_r', annot=True, fmt='d', square=True)

ax.set_xlabel('Predicted')

ax.set_ylabel('True');
predictions = [round(value) for value in y_adj_test]

pred_df = pd.DataFrame(predictions, columns=['prediction'])

pred_df['actual'] = y_test.reset_index(drop=True)

pred_df.head()
pred_df['actual'].sum()/pred_df['prediction'].sum()
(pred_df['actual'].sum()-pred_df['prediction'].sum())/pred_df['actual'].sum()*(1/pred_df.shape[0])*100