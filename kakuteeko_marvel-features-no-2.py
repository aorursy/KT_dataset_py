import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings("ignore")
%matplotlib inline
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv', encoding='cp949')
train = pd.merge(tr_train, y_train, how='left')
tr = pd.concat([tr_train, tr_test])
kfold = KFold(5, shuffle=True, random_state=0)
def make_data(features) :
    X_train = pd.DataFrame({'custid': tr_train.custid.unique()})
    X_test = pd.DataFrame({'custid': tr_test.custid.unique()})
    for f in features.values() :
        X_train = pd.merge(X_train, f, on='custid', how='left')
        X_test = pd.merge(X_test, f, on='custid', how='left')   
    IDtest = X_test.custid;
    X_train.drop(['custid'], axis=1, inplace=True)
    X_test.drop(['custid'], axis=1, inplace=True)
    for idx, val in enumerate(features.keys()) :
        print("적용된 Feature Method{} : {}".format(idx, val))
    return X_train, X_test
def makeBOW(col):
    f = lambda x: np.where(len(x) >=1, 1, 0)
    bow_data = pd.pivot_table(tr, index='custid', columns=col, values='tot_amt',
                             aggfunc=f, fill_value=0).reset_index()
    return bow_data
features = {}
def make_menrank_score(col, func, low_amt, size_list) :
    g = train.pivot_table('tot_amt', col,'gender', aggfunc=func).fillna(0).astype(int)
    g['men_rate'] = g[0] / g.sum(axis=1)
    g = g.sort_values(ascending=False, by='men_rate')
    f_sum = pd.DataFrame({"custid":tr.custid.unique(), ('men_score_'+ col):0 })
    for size in size_list :
        women_brd_list = g[g.sum(axis=1)>=low_amt].head(size).index
        tr['men_score_'+ col + str(size)] = tr[[col]].isin(women_brd_list).astype(int)
        f = tr.groupby('custid')[['men_score_'+ col + str(size)]].mean().reset_index()
        f_sum.iloc[:,1] = f_sum.iloc[:,1] + f.iloc[:,1]
    return f_sum
def make_womenrank_score(col, func, low_amt, size_list) :
    g = train.pivot_table('tot_amt', col,'gender', aggfunc=func).fillna(0).astype(int)
    g['women_rate'] = g[0] / g.sum(axis=1)
    g = g.sort_values(ascending=False, by='women_rate')
    f_sum = pd.DataFrame({"custid":tr.custid.unique(), ('women_score_'+ col):0 })
    for size in size_list :
        women_brd_list = g[g.sum(axis=1)>=low_amt].head(size).index
        tr['women_score_'+ col + str(size)] = tr[[col]].isin(women_brd_list).astype(int)
        f = tr.groupby('custid')[['women_score_'+ col + str(size)]].mean().reset_index()
        f_sum.iloc[:,1] = f_sum.iloc[:,1] + f.iloc[:,1]
    return f_sum
brand_gender_index = train.groupby("brd_nm")["gender"].mean().sort_values(ascending=False)
male_brd_list = brand_gender_index.loc[brand_gender_index>=0.9].index.tolist()
female_brd_list = brand_gender_index.loc[brand_gender_index<0.4].index.tolist()
tr['male_brd'] = tr.brd_nm.map(lambda x : 1 if x in male_brd_list else 0)
tr['female_brd'] = tr.brd_nm.map(lambda x : 1 if x in female_brd_list else 0)
f = tr.groupby('custid')['male_brd','female_brd'].mean().reset_index()
features['고성별연관제품구매여부'] = f;
X_train, X_test = make_data(features)
xgb = XGBClassifier(random_state=0)
score = cross_val_score(xgb, X_train, y_train.gender, cv=kfold, scoring="roc_auc", n_jobs=8)
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
features['여성제품구매점수count'] = make_womenrank_score('goodcd','count',15,range(10,3000,100))
features['남성제품구매점수count'] = make_menrank_score('goodcd','count',200000,range(10,3000,100))
features['여성브랜드구매점수count'] = make_womenrank_score('brd_nm','count',15,range(10,900,10))
features['남성브랜드구매점수count'] = make_menrank_score('brd_nm','count',200000,range(10,900,10))
features['여성코너구매점수count'] = make_womenrank_score('corner_nm','count',50,range(10,200,10))
features['남성코너구매점수count'] = make_menrank_score('corner_nm','count',200000,range(10,200,10))
f = tr.groupby('custid')['tot_amt'].agg([('평균구매액', 'mean')]).reset_index()
features['평균구매액'] = f;
tr['timeslot'] = tr.sales_time.map(lambda x : "{}시_구매건수".format(str(x)[:2]))
f = pd.pivot_table(tr, index='custid', columns='timeslot', values='tot_amt', 
                   aggfunc=np.size).fillna(0).astype(int).reset_index()
features['시간대별구매건수'] = f;
def f2(x):
    if 9 <= x <= 12 :
        return('아침_구매건수')
    elif 13 <= x <= 17 :
        return('점심_구매건수')
    else :
        return('저녁_구매건수')  # datatime 필드가 시간 형식에 맞지 않은 값을 갖는 경우 저녁시간으로 처리
    
tr['sales_hour'] = tr.sales_time.map(lambda x : np.int(str(x)[:2]))
tr['timeslot'] = tr['sales_hour'].apply(f2)
f = pd.pivot_table(tr, index='custid', columns='timeslot', values='tot_amt', 
                   aggfunc=np.size).fillna(0).astype(int).reset_index()
features['아침점심저녁구매건수'] = f;
f = tr.groupby('custid')['goodcd'].agg([('구매상품다양성', lambda x: len(x.unique()))]).reset_index()
f.구매상품다양성 = np.log(f.구매상품다양성)
features['구매상품다양성'] = f;
tr['sdate'] = tr.sales_date.str[:10]
f = tr.groupby(by = 'custid')['sdate'].agg([('내점일수','nunique')]).reset_index()
features['내점일수'] = f;
f = tr.pivot_table(index='custid', columns='pc_nm',values='brd_nm',aggfunc='count').fillna(0).astype(int)
f.columns = "pc_" + f.columns + "_구매건수"
f.reset_index(inplace=True)
features['pc_nm별구매건수'] = f;

f = tr.pivot_table(index='custid', columns='part_nm',values='brd_nm',aggfunc='count').fillna(0).astype(int)
f.columns = "part_" + f.columns + "_구매건수"
f.reset_index(inplace=True)
features['part_nm별구매수'] = f;

f = tr.pivot_table(index='custid', columns='team_nm',values='brd_nm',aggfunc='count').fillna(0).astype(int)
f.columns = "team_" + f.columns + "_구매건수"
f.reset_index(inplace=True)
features['team_nm별구매건수'] = f;

f = tr.pivot_table(index='custid', columns='buyer_nm',values='brd_nm',aggfunc='count').fillna(0).astype(int)
f.columns = "buyer_" + f.columns + "_구매건수"
f.reset_index(inplace=True)
features['buyer_nm별구매수'] = f;
X_train, X_test = make_data(features)
X_train.shape
log = LogisticRegression(random_state=0, C=100, penalty='l1')
score = cross_val_score(log, X_train, y_train.gender, cv=kfold, scoring="roc_auc", n_jobs=8)
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
xgb = XGBClassifier(random_state=0)
score = cross_val_score(xgb, X_train, y_train.gender, cv=kfold, scoring="roc_auc", n_jobs=8)
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
voting = VotingClassifier([('log',log), ('xgb', xgb)], voting='soft')

score = cross_val_score(voting, X_train, y_train.gender, cv=kfold, scoring="roc_auc", n_jobs=8)
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
IDtest = pd.DataFrame({"custid":tr_test.custid.unique()})
pred = xgb.fit(X_train, y_train.gender).predict_proba(X_test)[:,1]
fname = 'submission_voting.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))