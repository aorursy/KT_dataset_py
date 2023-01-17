import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 20)
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr = pd.concat([tr_train, tr_test])
tr
tr_train.columns
# 차원축소 매소드 
from sklearn.decomposition import PCA

def dummy_to_pca(tr, column_name:str, features) :
    max_seq = 300
    max_d = 15
    col_count = tr.groupby(column_name)[column_name].count()
    if len(col_count) > max_seq:
        tops = col_count.sort_values(ascending=False)[0:max_seq].index
        f =tr.loc[tr[column_name].isin(tops)][['custid', column_name]]
    else:
        tops = col_count.index
        f =tr[['custid', column_name]]
    f = pd.get_dummies(f, columns=[column_name])  # This method performs One-hot-encoding
    f = f.groupby('custid').mean()
    if len(tops) < max_d:
        max_d = len(tops)
    pca = PCA(n_components=max_d)
    pca.fit(f)
    cumsum = np.cumsum(pca.explained_variance_ratio_) #분산의 설명량을 누적합
    #print(cumsum)
    num_d = np.argmax(cumsum >= 0.99) + 1 # 분산의 설명량이 99%이상 되는 차원의 수
    if num_d == 1:
        num_d = max_d
    pca = PCA(n_components=num_d)    
    result = pca.fit_transform(f)
    result = pd.DataFrame(result)
    result.columns = [column_name + '_' + str(column) for column in result.columns]
    result.index = f.index
    return pd.concat([features, result], axis=1, join_axes=[features.index])
#f = dummy_to_pca(tr, 'team_nm', f)
#f
# 학습을 위한 데이터를 만든다.
f = tr.groupby('custid').agg({
    'tot_amt': [('총구매액', 'sum'),('구매건수', 'size'),('평균구매가격', 'mean'),('최대구매액', 'max')],
    'dis_amt': [('dis_sum', 'sum'),('dis_mean', 'mean')],
    'net_amt': [('net_sum', 'sum'),('net_mean', 'mean')],
    'inst_mon': [('평균할부개월수', 'mean'), ('최대할부개월수', 'max')],
    'brd_nm': [('구매상품다양성', lambda x: x.nunique()), 
               ('구매상품다양성비', lambda x: x.nunique()/x.count())],
    'import_flg': [('수입상품_구매비율', "mean"), ('수입상품_구매수', 'sum')],
    'sales_date': [
        ('내점일수',lambda x: x.str[:10].nunique()),
        ('내점비율',lambda x: x.str[:10].nunique()/x.count()),
        ('주말방문비율', lambda x: np.mean(pd.to_datetime(x).dt.dayofweek>4)),
        ('봄-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([3,4,5]))),
        ('여름-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([6,7,8]))),
        ('가을-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([9,10,11]))),
        ('겨울-구매비율', lambda x: np.mean( pd.to_datetime(x).dt.month.isin([1,2,12])))
    ],
    'sales_time': [('밤구입비율', lambda x: np.count_nonzero(x.astype(np.int)[(x>1800)|(x<900)])/ x.count())],
    }).reset_index()
f.columns = f.columns.get_level_values(1)
f.rename(columns={'': 'custid'}, inplace=True)
f = dummy_to_pca(tr, 'brd_nm', f)
f = dummy_to_pca(tr, 'corner_nm', f)
f = dummy_to_pca(tr, 'pc_nm', f)
f = dummy_to_pca(tr, 'part_nm', f)
f = dummy_to_pca(tr, 'buyer_nm', f)
f = dummy_to_pca(tr, 'team_nm', f)
f = dummy_to_pca(tr, 'goodcd', f)
f = dummy_to_pca(tr, 'str_nm', f)
tr['month'] = pd.to_datetime(tr['sales_date']).dt.month.astype(str)
f = dummy_to_pca(tr, 'month', f)
tr['week'] = pd.to_datetime(tr['sales_date']).dt.dayofweek.astype(str)
f = dummy_to_pca(tr, 'week', f)
tr['time'] = np.floor(tr['sales_time']/100).astype(int).astype(str)
f = dummy_to_pca(tr, 'time', f)
f
# 데이터를 분리한다.
X_train = pd.DataFrame({'custid': tr_train.custid.unique()})
X_train = pd.merge(X_train, f, how='left')
display(X_train)

X_test = pd.DataFrame({'custid': tr_test.custid.unique()})
X_test = pd.merge(X_test, f, how='left')
display(X_test)
IDtest = X_test.custid;
X_train.drop(['custid'], axis=1, inplace=True)
X_test.drop(['custid'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
# 학습한다.
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#parameters = {'max_depth': 6, 'n_estimators': 200}
#clf = RandomForestClassifier(**parameters, random_state=0)

#parameters = {'xgb__max_depth': 3, 'xgb__subsample': 0.7}
parameters = {'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 1.0, 'learning_rate': 0.05, 
              'min_child_weight': 5, 'silent': True, 'n_estimators': 200}
clf = XGBClassifier(**parameters, random_state=0, n_jobs=-1)
# clf = XGBClassifier()
# parameters = {
#     'max_depth': [4],
#     'subsample': [0.9],
#     'colsample_bytree': [1.0],
#     'learning_rate' : [0.05],
#     'min_child_weight': [5],
#     'silent': [True],
#     'n_estimators': [200]
# }
# clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
# clf.fit(X_train, y_train)
# best_est = clf.best_estimator_
# print(best_est)
score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')

print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
pred = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
fname = 'submission.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))