%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import datetime
import itertools
df_ks = pd.read_csv("../input/ks-projects-201801.csv")

display(df_ks.head())
df_ks.describe()
df_state = df_ks.groupby('state').count()
display(df_state)
df_main_category = df_ks.groupby('main_category')
df_main_category = df_main_category['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_main_category)
df_main_category.describe()
df_main_category.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_category = df_ks.groupby('category')
df_category = df_category['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_category)
df_category.describe()
df_category.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(30,5))
plt.legend(loc='upper left')
plt.show()
df_currency = df_ks.groupby('currency')
df_currency = df_currency['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_currency)
df_currency.describe()
df_currency.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_country = df_ks.groupby('country')
df_country = df_country['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_country)
df_country.describe()
df_country.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
#launchedとdeadlineから期間を算出しtermとして追加
str2date = lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d')
df_ks['term'] = df_ks['deadline'].map(str2date)-df_ks['launched'].map(str2date)
df_ks['term'] = df_ks['term'].map(lambda x: x.days)
df_term = df_ks.groupby('term')
df_term = df_term['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_term)
df_term.describe()
df_term.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
#launched, deadlineのymdをlaunched_y,launched_m,launched_d,deadline_y,deadline_m,deadline_dとして列追加
df_ks['launched_y'] = df_ks['launched'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').year)
df_ks['launched_m'] = df_ks['launched'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').month)
df_ks['launched_d'] = df_ks['launched'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').day)
df_ks['deadline_y'] = df_ks['deadline'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').year)
df_ks['deadline_m'] = df_ks['deadline'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').month)
df_ks['deadline_d'] = df_ks['deadline'].map(lambda x :datetime.datetime.strptime(x[:10], '%Y-%m-%d').day)
df_ks.head()
df_launched_y = df_ks.groupby('launched_y')
df_launched_y = df_launched_y['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_y)
df_launched_y.describe()
df_launched_y.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='best')
plt.show()
df_launched_m = df_ks.groupby('launched_m')
df_launched_m = df_launched_m['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_m)
df_launched_m.describe()
df_launched_m.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='best')
plt.show()
df_launched_d = df_ks.groupby('launched_d')
df_launched_d = df_launched_d['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_d)
df_launched_d.describe()
df_launched_d.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_y = df_ks.groupby('deadline_y')
df_deadline_y = df_deadline_y['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_y)
df_deadline_y.describe()
df_deadline_y.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_m = df_ks.groupby('deadline_m')
df_deadline_m = df_deadline_m['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_m)
df_deadline_m.describe()
df_deadline_m.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_d = df_ks.groupby('deadline_d')
df_deadline_d = df_deadline_d['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_d)
df_deadline_d.describe()
df_deadline_d.plot.bar(y=['successful','failed','live','canceled','suspended','undefined'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
#stateごとにgoalの値がどうなっているかを確認
df_state = df_ks.groupby('state')
df_state['usd_goal_real'].describe()
df_ks_undefined = df_ks[df_ks['state'] == 'undefined']
df_ks_undefined['state'].where(df_ks_undefined['usd_pledged_real'] < df_ks_undefined['usd_goal_real'], 'successful',inplace=True)
df_ks_undefined['state'].where(df_ks_undefined['usd_pledged_real'] >= df_ks_undefined['usd_goal_real'], 'failed',inplace=True)
display(df_ks_undefined)
df_ks = df_ks[(df_ks['state'] != 'live') & (df_ks['state'] != 'undefined')]
df_ks = df_ks[df_ks['term'] < 14709] 
df_ks = pd.concat([df_ks, df_ks_undefined])

df_state = df_ks.groupby('state').count()
display(df_state)
df_category = df_ks.groupby('category')
df_category = df_category['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_category)
df_category.describe()
df_category.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(60,5))
plt.legend(loc='upper left')
plt.show()
df_main_category = df_ks.groupby('main_category')
df_main_category = df_main_category['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_main_category)
df_main_category.describe()
df_main_category.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_currency = df_ks.groupby('currency')
df_currency = df_currency['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_currency)
df_currency.describe()
df_currency.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_country = df_ks.groupby('country')
df_country = df_country['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_country)
df_country.describe()
df_country.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_term = df_ks.groupby('term')
df_term = df_term['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_term)
df_term.describe()
df_term.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(20,5))
plt.legend(loc='upper left')
plt.show()
df_launched_y = df_ks.groupby('launched_y')
df_launched_y = df_launched_y['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_y)
df_launched_y.describe()
df_launched_y.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_launched_m = df_ks.groupby('launched_m')
df_launched_m = df_launched_m['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_m)
df_launched_m.describe()
df_launched_m.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_launched_d = df_ks.groupby('launched_d')
df_launched_d = df_launched_d['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_launched_d)
df_launched_d.describe()
df_launched_d.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_y = df_ks.groupby('deadline_y')
df_deadline_y = df_deadline_y['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_y)
df_deadline_y.describe()
df_deadline_y.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_m = df_ks.groupby('deadline_m')
df_deadline_m = df_deadline_m['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_m)
df_deadline_m.describe()
df_deadline_m.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_deadline_d = df_ks.groupby('deadline_d')
df_deadline_d = df_deadline_d['state'].value_counts(normalize=True).unstack(fill_value=0)
display(df_deadline_d)
df_deadline_d.describe()
df_deadline_d.plot.bar(y=['successful','failed','canceled','suspended'], stacked=True, figsize=(15,5))
plt.legend(loc='upper left')
plt.show()
df_ks['category_rate'] = df_ks['category'].replace(df_category['successful'])
df_ks['main_category_rate'] = df_ks['main_category'].replace(df_main_category['successful'])
df_ks['currency_rate'] = df_ks['currency'].replace(df_currency['successful'])
df_ks['country_rate'] = df_ks['country'].replace(df_country['successful'])
df_ks['launched_y_rate'] = df_ks['launched_y'].replace(df_launched_y['successful'])
df_ks['launched_m_rate'] = df_ks['launched_m'].replace(df_launched_m['successful'])
df_ks['launched_d_rate'] = df_ks['launched_d'].replace(df_launched_d['successful'])
df_ks['deadline_y_rate'] = df_ks['deadline_y'].replace(df_deadline_y['successful'])
df_ks['deadline_m_rate'] = df_ks['deadline_m'].replace(df_deadline_m['successful'])
df_ks['deadline_d_rate'] = df_ks['deadline_d'].replace(df_deadline_d['successful'])
df_ks.head()
mms = MinMaxScaler()
df_ks['usd_goal_real_mms'] = mms.fit_transform(df_ks[['usd_goal_real']].values)
df_ks['category_rate_mms'] = mms.fit_transform(df_ks[['category_rate']].values)
df_ks['main_category_rate_mms'] = mms.fit_transform(df_ks[['main_category_rate']].values)
df_ks['currency_rate_mms'] = mms.fit_transform(df_ks[['currency_rate']].values)
df_ks['country_rate_mms'] = mms.fit_transform(df_ks[['country_rate']].values)
df_ks['term_mms'] = mms.fit_transform(df_ks[['term']].values)
df_ks['launched_y_mms'] = mms.fit_transform(df_ks[['launched_y_rate']].values)
df_ks['launched_m_mms'] = mms.fit_transform(df_ks[['launched_m_rate']].values)
df_ks['launched_d_mms'] = mms.fit_transform(df_ks[['launched_d_rate']].values)
df_ks['deadline_y_mms'] = mms.fit_transform(df_ks[['deadline_y_rate']].values)
df_ks['deadline_m_mms'] = mms.fit_transform(df_ks[['deadline_m_rate']].values)
df_ks['deadline_d_mms'] = mms.fit_transform(df_ks[['deadline_d_rate']].values)
df_ks.head()
#stateがsuccessfulとそれ以外のもので分割したものを用意
df_ks_s = df_ks[df_ks['state'] == "successful"]
df_ks_f = df_ks[df_ks['state'] != "successful"]
df_ks_s.describe()
list_expl = ['term_mms','main_category_rate_mms','country_rate_mms',\
            'launched_y_mms','launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms','usd_goal_real_mms']
listiter = list(itertools.combinations(list_expl, 2))
for x_v, y_v in listiter:
    x_suc = df_ks_s[x_v].values
    y_suc = df_ks_s[y_v].values
    x_fail = df_ks_f[x_v].values
    y_fail = df_ks_f[y_v].values

    plt.plot(x_fail,y_fail, '.', color='red', alpha=0.2, label='failed')
    plt.plot(x_suc, y_suc, '.', color='blue', alpha=0.2, label='successful')
    plt.grid(which='major',color='black',linestyle=':')
    plt.grid(which='minor',color='black',linestyle=':')
    plt.xlabel(x_v)
    plt.ylabel(y_v)
    plt.legend(loc='best')

    plt.show()
#対数軸で拡大してみる
x_suc = df_ks_s['term_mms'].values
y_suc = df_ks_s['usd_goal_real_mms'].values
x_fail = df_ks_f['term_mms'].values
y_fail = df_ks_f['usd_goal_real_mms'].values

plt.plot(x_fail,y_fail, '.', color='red', alpha=0.2, label='failed')
plt.plot(x_suc, y_suc,  '.', color='blue', alpha=0.2, label='successful')
plt.grid(which='major',color='black',linestyle=':')
plt.grid(which='minor',color='black',linestyle=':')
plt.xlabel("term")
plt.ylabel("log(usd_goal_real)")
plt.yscale('log')
plt.legend(loc='best')

plt.show()
#main_category別に確認
df_ks_s = df_ks[df_ks['state'] == "successful"]
df_ks_f = df_ks[df_ks['state'] == "failed"]

mc = list(set(df_ks['main_category'].values))

x_s_cat = [0 for _ in range(len(mc))]
y_s_cat = [0 for _ in range(len(mc))]
x_f_cat = [0 for _ in range(len(mc))]
y_f_cat = [0 for _ in range(len(mc))]

for i in range(len(mc)):
    x_s_cat[i] = df_ks_s[df_ks_s['main_category'] == mc[i]]['term_mms'].values
    y_s_cat[i] = df_ks_s[df_ks_s['main_category'] == mc[i]]['usd_goal_real_mms'].values
    x_f_cat[i] = df_ks_f[df_ks_f['main_category'] == mc[i]]['term_mms'].values
    y_f_cat[i] = df_ks_f[df_ks_f['main_category'] == mc[i]]['usd_goal_real_mms'].values

    plt.title("main_category : " + mc[i])
    plt.plot(x_f_cat[i], y_f_cat[i], '.', alpha=0.2, color='red', label='failed')
    plt.plot(x_s_cat[i], y_s_cat[i], '.', alpha=0.2, color='blue', label='successful')
    plt.grid(which='major',color='black',linestyle=':')
#    plt.grid(which='minor',color='black',linestyle=':')
    plt.xlabel("term")
    plt.ylabel("log(usd_goal_real)")
    plt.yscale('log')
    plt.legend(loc='best')
    
    plt.show()
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf = SGDClassifier(loss='log', penalty='L2', max_iter=5000, fit_intercept=True, random_state=1234)
clf.fit(X_train, y_train)

w = [0 for _ in range(len(expl_val)+1)]
w[0] = clf.intercept_[0]
s = "w0 = {:.3f}  ".format(w[0])
for i in range(len(expl_val)):
    w[i+1] = clf.coef_[0, i]
    s += "w" + str(i+1) + (" = {:.3f}  ".format(w[i+1]))
print(s)
y_pred_train = clf.predict(X_train)
print('対数尤度 = {:.3f}'.format(log_loss(y_train, y_pred_train)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_pred_train)))
precision, recall, f1_score, _ = precision_recall_fscore_support(y_train, y_pred_train)

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
conf_mat = pd.DataFrame(confusion_matrix(y_train, y_pred_train), 
                        index=['正解 = failed', '正解 = successful'], 
                        columns=['予測 = failed', '予測 = successful'])
conf_mat
y_pred_test = clf.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred_test)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))
precision2, recall2, f1_score2, _ = precision_recall_fscore_support(y_test, y_pred_test)

print('適合率（Precision） = {:.3f}%'.format(100 * precision2[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall2[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score2[0]))
conf_mat2 = pd.DataFrame(confusion_matrix(y_test, y_pred_test), 
                        index=['正解 = failed', '正解 = successful'], 
                        columns=['予測 = failed', '予測 = successful'])
conf_mat2
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
estimator = SGDClassifier(loss='log', penalty='none', max_iter=500, fit_intercept=True, random_state=1234)
rfecv = RFECV(estimator, cv=5, scoring='accuracy')
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

rfecv.fit(X, y)
print('Feature ranking: \n{}'.format(rfecv.ranking_))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
estimator = LassoCV(normalize=True, cv=10)
sfm = SelectFromModel(estimator, threshold=1e-5)
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

sfm.fit(X, y)
abs_coef = np.abs(sfm.estimator_.coef_)
print(abs_coef)
plt.barh(np.arange(0, len(abs_coef)), abs_coef, tick_label=expl_val)
plt.show()
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf = SGDClassifier(loss='log', penalty='none', max_iter=5000, fit_intercept=True, random_state=1234)
clf.fit(X_train, y_train)

w = [0 for _ in range(len(expl_val)+1)]
w[0] = clf.intercept_[0]
s = "w0 = {:.3f}  ".format(w[0])
for i in range(len(expl_val)):
    w[i+1] = clf.coef_[0, i]
    s += "w" + str(i+1) + (" = {:.3f}  ".format(w[i+1]))
print(s)
y_pred_train = clf.predict(X_train)
print('対数尤度 = {:.3f}'.format(log_loss(y_train, y_pred_train)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_pred_train)))
y_pred_test = clf.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred_test)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms','launched_y_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf = SGDClassifier(loss='log', penalty='none', max_iter=5000, fit_intercept=True, random_state=1234)
clf.fit(X_train, y_train)

w = [0 for _ in range(len(expl_val)+1)]
w[0] = clf.intercept_[0]
s = "w0 = {:.3f}  ".format(w[0])
for i in range(len(expl_val)):
    w[i+1] = clf.coef_[0, i]
    s += "w" + str(i+1) + (" = {:.3f}  ".format(w[i+1]))
print(s)
y_pred_train = clf.predict(X_train)
print('対数尤度 = {:.3f}'.format(log_loss(y_train, y_pred_train)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_train, y_pred_train)))
y_pred_test = clf.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred_test)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred_test)))
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms','launched_y_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(X, y, test_size=test_size, random_state=1234)
%%time
#param = [{'C': np.array(2.0)**list(range(0, 4)), 'kernel': ['rbf'], 'gamma': np.array(2.0)**list(range(-2, 2))}]
#gs = GridSearchCV(SVC(random_state=1234, shrinking=False, verbose=1), param, cv=5,)
#gs.fit(X_train_cv, y_train_cv)
#print(gs.best_params_, gs.best_score_)
%%time
#C_, gamma_, _ = gs.best_params_.values()
C_, gamma_ = 2, 2
clf2 = SVC(C=C_, gamma=gamma_, kernel='rbf', shrinking=False, random_state=1234)
clf2.fit(X_train_cv, y_train_cv)
%%time
y_pred = clf2.predict(X_train_cv)
print('対数尤度 = {:.3f}'.format(log_loss(y_train_cv, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_train_cv, y_pred)))
precision, recall, f1_score, _ = precision_recall_fscore_support(y_train_cv, y_pred)

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
conf_mat = pd.DataFrame(confusion_matrix(y_train_cv, y_pred), 
                        index=['正解 = failed', '正解 = successful'], 
                        columns=['予測 = failed', '予測 = successful'])
conf_mat
%%time
y_pred = clf2.predict(X_test_final)
print('対数尤度 = {:.3f}'.format(log_loss(y_test_final, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test_final, y_pred)))
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_final, y_pred)

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
conf_mat = pd.DataFrame(confusion_matrix(y_test_final, y_pred), 
                        index=['正解 = failed', '正解 = successful'], 
                        columns=['予測 = failed', '予測 = successful'])
conf_mat
%%time
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from IPython.display import Image

clf3 = DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_leaf=3, random_state=1234)

expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf3 = clf3.fit(X_train, y_train)
print("score=", clf3.score(X_train, y_train))

print(clf3.feature_importances_)
pd.DataFrame(clf3.feature_importances_, index=expl_val).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred = clf3.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred)))
%%time
from sklearn.ensemble import RandomForestClassifier

clf4 = RandomForestClassifier(n_estimators=10, max_depth=3, criterion="gini", min_samples_leaf=2, min_samples_split=2, random_state=1234)

expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf4 = clf4.fit(X_train, y_train)
print("score=", clf4.score(X_train, y_train))

print(clf4.feature_importances_)
pd.DataFrame(clf4.feature_importances_, index=expl_val).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred = clf4.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred)))
%%time
expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','deadline_y_mms','deadline_m_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf4 = clf4.fit(X_train, y_train)
print("score=", clf4.score(X_train, y_train))

print(clf4.feature_importances_)
pd.DataFrame(clf4.feature_importances_, index=expl_val).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred = clf4.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred)))
# 決定木の描画
for i, est in enumerate(clf4.estimators_):
    export_graphviz(est, out_file='rf'+str(i)+'.dot', feature_names=expl_val, class_names=["falied","successed"], filled=True, rounded=True, special_characters=True)
!dot -Tpng rf0.dot -o rf0.png -Gdpi=100
!dot -Tpng rf1.dot -o rf1.png -Gdpi=100
!dot -Tpng rf2.dot -o rf2.png -Gdpi=100
!dot -Tpng rf3.dot -o rf3.png -Gdpi=100
!dot -Tpng rf4.dot -o rf4.png -Gdpi=100
!dot -Tpng rf5.dot -o rf5.png -Gdpi=100
!dot -Tpng rf6.dot -o rf6.png -Gdpi=100
!dot -Tpng rf7.dot -o rf7.png -Gdpi=100
!dot -Tpng rf8.dot -o rf8.png -Gdpi=100
!dot -Tpng rf9.dot -o rf9.png -Gdpi=100
Image(filename = 'rf0.png')
Image(filename = 'rf1.png')
Image(filename = 'rf2.png')
Image(filename = 'rf3.png')
Image(filename = 'rf4.png')
Image(filename = 'rf5.png')
Image(filename = 'rf6.png')
Image(filename = 'rf7.png')
Image(filename = 'rf8.png')
Image(filename = 'rf9.png')
%%time
from sklearn.ensemble import AdaBoostClassifier
clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, min_samples_split=2, random_state=1234, criterion="gini"), n_estimators=10, random_state=1234)

expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms','deadline_d_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf5 = clf5.fit(X_train, y_train)
print("score=", clf5.score(X_train, y_train))

print(clf5.feature_importances_)
pd.DataFrame(clf5.feature_importances_, index=expl_val).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred = clf5.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred)))
%%time
clf5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, min_samples_split=2, random_state=1234, criterion="gini"), n_estimators=10, random_state=1234)

expl_val = ['term_mms','usd_goal_real_mms','category_rate_mms','country_rate_mms',\
            'launched_y_mms', 'launched_m_mms','launched_d_mms','deadline_y_mms','deadline_m_mms']

y = np.where((df_ks['state'] == "successful").values, 1, 0)
X = df_ks[expl_val].values

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)

clf5 = clf5.fit(X_train, y_train)
print("score=", clf5.score(X_train, y_train))

print(clf5.feature_importances_)
pd.DataFrame(clf5.feature_importances_, index=expl_val).plot.bar(figsize=(7,2))
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()
y_pred = clf5.predict(X_test)
print('対数尤度 = {:.3f}'.format(log_loss(y_test, y_pred)))
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_pred)))
# 決定木の描画
for i, est in enumerate(clf5.estimators_):
    export_graphviz(est, out_file='ab'+str(i)+'.dot', feature_names=expl_val, class_names=["falied","successed"], filled=True, rounded=True, special_characters=True)
!dot -Tpng ab0.dot -o ab0.png -Gdpi=100
!dot -Tpng ab1.dot -o ab1.png -Gdpi=100
!dot -Tpng ab2.dot -o ab2.png -Gdpi=100
!dot -Tpng ab3.dot -o ab3.png -Gdpi=100
!dot -Tpng ab4.dot -o ab4.png -Gdpi=100
!dot -Tpng ab5.dot -o ab5.png -Gdpi=100
!dot -Tpng ab6.dot -o ab6.png -Gdpi=100
!dot -Tpng ab7.dot -o ab7.png -Gdpi=100
!dot -Tpng ab8.dot -o ab8.png -Gdpi=100
!dot -Tpng ab9.dot -o ab9.png -Gdpi=100
Image(filename = 'ab0.png')
Image(filename = 'ab1.png')
Image(filename = 'ab2.png')
Image(filename = 'ab3.png')
Image(filename = 'ab4.png')
Image(filename = 'ab5.png')
Image(filename = 'ab6.png')
Image(filename = 'ab7.png')
Image(filename = 'ab8.png')
Image(filename = 'ab9.png')
