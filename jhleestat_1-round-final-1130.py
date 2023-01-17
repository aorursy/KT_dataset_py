import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 20)
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr = pd.concat([tr_train, tr_test])
tr
features = []
f = tr.groupby('custid')['tot_amt'].agg([('총구매액', 'sum')]).reset_index()
features.append(f); f
f = tr.groupby('custid')['tot_amt'].agg([('구매건수', 'size')]).reset_index()
features.append(f); f
f = tr.groupby('custid')['tot_amt'].agg([('평균구매가격', 'mean')]).reset_index()
features.append(f); f
f = tr.groupby('custid')['inst_mon'].agg([('평균할부개월수', 'mean')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f); f
#n = tr.part_nm.nunique()
#f = tr.groupby('custid')['brd_nm'].agg([('구매상품다양성', lambda x: len(x.unique()) / n)]).reset_index()
#features.append(f); f
#n = 4
#f = tr.groupby('custid')['str_nm'].agg([('매장이용다양성', lambda x: len(x.unique()) / n)]).reset_index()
#features.append(f); f
tr['sdate'] = tr.sales_date.str[:10]
f = tr.groupby(by = 'custid')['sdate'].agg([('내점일수','nunique')]).reset_index()
features.append(f); f
x = tr[tr['import_flg'] == 1].groupby('custid').size() / tr.groupby('custid').size()
f = x.reset_index().rename(columns={0: '수입상품_구매비율'}).fillna(0)
f.iloc[:,1] = (f.iloc[:,1]*100).apply(round, args=(1,))
features.append(f); f
def f2(x):
    k = x.dayofweek
    if k <= 2 :
        return('월화수_구매건수')
    elif 3 <= k < 5 :
        return('목금_구매건수')
    elif 5 <= k < 6 :
        return('토_구매건수')
    else :
        return('일_구매건수')    
    
tr['요일2'] = pd.to_datetime(tr.sales_date).apply(f2)
f = pd.pivot_table(tr, index='custid', columns='요일2', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
#def fw(x):
#    k = x.dayofweek
#    if k <= 4 :
#        return('주중_방문')
#    else :
#        return('주말_방문')    
    
#df = tr.copy()
#df = df.drop_duplicates(['custid','sales_date'])

#df['week'] = pd.to_datetime(df.sales_date).apply(fw)
#df = pd.pivot_table(df, index='custid', columns='week', values='tot_amt', 
                   #aggfunc=np.size, fill_value=0).reset_index()
#df['주말방문비율'] = ((df.iloc[:,1] / (df.iloc[:,1]+df.iloc[:,2]))*100).apply(round, args=(1,))
#f = df.copy().iloc[:,[0,-1]]
#features.append(f); f
def f1(x):
    k = x.month
    if 2 <= k <= 4 :
        return('234월_구매건수')
    elif 5 <= k <= 7 :
        return('567월_구매건수')
    elif 8 <= k <= 10 :
        return('8910월_구매건수')
    else :
        return('11121월_구매건수')    
    
tr['season2'] = pd.to_datetime(tr.sales_date).apply(f1)
f = pd.pivot_table(tr, index='custid', columns='season2', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
def f2(x):
    if 901 <= x < 1200 :
        return('12시 이전_구매건수')
    elif 1200 <= x < 1400 :
        return('12~2시_구매건수')
    elif 1400 <= x < 1600 :
        return('2~4시_구매건수')
    elif 1600 <= x < 1800 :
        return('4~6시_구매건수')
    else :
        return('6시이후_구매건수')  

tr['timeslot2'] = tr.sales_time.apply(f2)
tr['timeslot2']
f = pd.pivot_table(tr, index='custid', columns='timeslot2', values='tot_amt',
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
f = tr.groupby('custid')['dis_amt'].agg([('평균할인금액', 'mean')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f); f
f = tr.groupby('custid')['sales_time'].agg([('평균구매시간', 'mean')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f); f
f = tr.groupby('custid')['net_amt'].agg([('실제구매금액', 'sum')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f); f
f = tr.groupby('custid')['net_amt'].agg([('실제구매금액평균', 'mean')]).reset_index()
f.iloc[:,1] = f.iloc[:,1].apply(round, args=(1,))
features.append(f); f
f = pd.pivot_table(tr, index='custid', columns='str_nm', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
f = pd.pivot_table(tr, index='custid', columns='part_nm', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
f = pd.pivot_table(tr, index='custid', columns='buyer_nm', values='tot_amt', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
X_train = pd.DataFrame({'custid': tr_train.custid.unique()})
for f in features :
    X_train = pd.merge(X_train, f, how='left')
display(X_train)

X_test = pd.DataFrame({'custid': tr_test.custid.unique()})
for f in features :
    X_test = pd.merge(X_test, f, how='left')
display(X_test)
X_train["평균할인율"] = (X_train["평균할인금액"] / X_train["평균구매가격"])*100
X_test["평균할인율"] = (X_test["평균할인금액"] / X_test["평균구매가격"])*100
X_train["토_비율"] = (X_train["토_구매건수"] / X_train["구매건수"])*100
X_train["일_비율"] = (X_train["일_구매건수"] / X_train["구매건수"])*100
X_test["토_비율"] = (X_test["토_구매건수"] / X_test["구매건수"])*100
X_test["일_비율"] = (X_test["일_구매건수"] / X_test["구매건수"])*100
X_train['가정용품R']=X_train['가정용품']+X_train['가정용품파트']
X_train['공산품R']=X_train['공산품']+X_train['공산품파트']
X_train['로얄부띠끄R']=X_train['로얄부띠끄']+X_train['로얄부틱']
X_train['생식품R']=X_train['생식품']+X_train['생식품파트']
X_train['스포츠캐주얼R']=X_train['스포츠캐주얼']+X_train['스포츠캐쥬얼']
X_train['여성캐주얼R']=X_train['여성캐주얼']+X_train['여성캐쥬얼']
X_train['잡화R']=X_train['잡화']+X_train['잡화파트']

X_test['가정용품R']=X_test['가정용품']+X_test['가정용품파트']
X_test['공산품R']=X_test['공산품']+X_test['공산품파트']
X_test['로얄부띠끄R']=X_test['로얄부띠끄']+X_test['로얄부틱']
X_test['생식품R']=X_test['생식품']+X_test['생식품파트']
X_test['스포츠캐주얼R']=X_test['스포츠캐주얼']+X_test['스포츠캐쥬얼']
X_test['여성캐주얼R']=X_test['여성캐주얼']+X_test['여성캐쥬얼']
X_test['잡화R']=X_test['잡화']+X_test['잡화파트']
X_train['남성파트']=X_train['가정용품R']+X_train['공산품R']+X_train['생식품R']+X_train['케주얼,구두,아동']
X_test['남성파트']=X_test['가정용품R']+X_test['공산품R']+X_test['생식품R']+X_test['케주얼,구두,아동']
X_train['여성파트']=X_train['여성캐주얼']+X_train['영캐릭터']+X_train['영플라자']+X_train['패션잡화'] 
X_test['여성파트']=X_test['여성캐주얼']+X_test['영캐릭터']+X_test['영플라자']+X_test['패션잡화'] 
#def f1(x):
#    if  x < 1200 :
#        return('1')
#    elif 1200 <= x < 1400 :
#        return('2')
#    elif 1400 <= x < 1700 :
#        return('3')
#    elif 1700 <= x < 1800 :
#        return('4')
#    else :
#        return('5')  

#X_train['평균구매시간B'] = X_train.평균구매시간.apply(f1)
#def f1(x):
#    if  x < 1200 :
#        return('1')
#    elif 1200 <= x < 1400 :
#        return('2')
#    elif 1400 <= x < 1700 :
#        return('3')
#    elif 1700 <= x < 1800 :
#        return('4')
#    else :
#        return('5')  

#X_test['평균구매시간B'] = X_test.평균구매시간.apply(f1)
f = X_train.총구매액.where(X_train.총구매액>=0, other=0)
f = np.log(f+1)
X_train.총구매액 = f

f = X_test.총구매액.where(X_test.총구매액>=0, other=0)
f = np.log(f+1)
X_test.총구매액 = f

f = X_train.평균구매가격.where(X_train.평균구매가격>=0, other=0)
f = np.log(f+1)
X_train.평균구매가격 = f

f = X_test.평균구매가격.where(X_test.평균구매가격>=0, other=0)
f = np.log(f+1)
X_test.평균구매가격 = f

f = X_train.구매건수.where(X_train.구매건수>=0, other=0)
f = np.log(f+1)
X_train.구매건수 = f

f = X_test.구매건수.where(X_test.구매건수>=0, other=0)
f = np.log(f+1)
X_test.구매건수 = f

f = X_train.내점일수.where(X_train.내점일수>=0, other=0)
f = np.log(f+1)
X_train.내점일수 = f

f = X_test.내점일수.where(X_test.내점일수>=0, other=0)
f = np.log(f+1)
X_test.내점일수 = f


f = X_train.실제구매금액평균.where(X_train.실제구매금액평균>=0, other=0)
f = np.log(f+1)
X_train.실제구매금액평균 = f

f = X_test.실제구매금액평균.where(X_test.실제구매금액평균>=0, other=0)
f = np.log(f+1)
X_test.실제구매금액평균 = f

f = X_train.실제구매금액.where(X_train.실제구매금액>=0, other=0)
f = np.log(f+1)
X_train.실제구매금액 = f

f = X_test.실제구매금액.where(X_test.실제구매금액>=0, other=0)
f = np.log(f+1)
X_test.실제구매금액 = f

IDtest = X_test.custid;
X_train.drop(['custid'], axis=1, inplace=True)
X_test.drop(['custid'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
X_test.info()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=2)
score = cross_val_score(gbc, X_train, y_train, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
from xgboost import XGBClassifier
parameters = {'xgb__max_depth': 14, 'xgb__subsample': 0.4}
clf = XGBClassifier(**parameters, random_state=0, n_jobs=-1)
score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('gbc', gbc),('clf', clf)], voting='soft')
score = cross_val_score(votingC, X_train, y_train, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
pred = votingC.fit(X_train, y_train).predict_proba(X_test)[:,1]
fname = 'submission1130.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))