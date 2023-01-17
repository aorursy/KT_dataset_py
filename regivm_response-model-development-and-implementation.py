import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/promoted.csv")
df.head(3)
df.info()
df.describe()
df['card_tenure'].fillna(df['card_tenure'].mean(), inplace=True)
df['avg_bal'].fillna(df['avg_bal'].mean(), inplace=True)
df.describe()
df['geo_group'].value_counts(dropna=False)
df['res_type'].value_counts(dropna=False)
df['geo_group'].fillna('E', inplace=True)
df['res_type'].fillna('CO', inplace=True)
df.info()
dfg=pd.get_dummies(df['geo_group'], prefix='geo', drop_first=True)
dfr=pd.get_dummies(df['res_type'], prefix='res', drop_first=True)
dfrg=df.join([dfg,dfr])
dfrg.drop(['geo_group', 'res_type'], axis=1, inplace=True)
df=dfrg
df.head(3)

y=df['resp']
X=df.iloc[:, 2:]
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=11)
#X_train.head(3)
y_train.mean()
import statsmodels.api as sm
from pandas.core import datetools

xc = sm.add_constant(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() 
vif["vif"] = [variance_inflation_factor(xc.values, i) for i in range(xc.shape[1])]
vif["features"] = xc.columns
vif
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())
y=df['resp']
X=df.iloc[:, 2:]

X.drop(['res_CO','res_RE', 'res_TO','geo_N', 'geo_W','num_promoted', 'geo_SE', 'res_SI'], axis=1, inplace=True)

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=11)


import statsmodels.api as sm
from pandas.core import datetools

xc = sm.add_constant(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() 
vif["vif"] = [variance_inflation_factor(xc.values, i) for i in range(xc.shape[1])]
vif["features"] = xc.columns
print(vif)

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary())
import statsmodels.api as sm
logit_model=sm.Logit(y_test,X_test)
result=logit_model.fit()
print(result.summary())
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

prb=logreg.predict_proba(X_train)[:,1]
type(prb)
#pd.DataFrame(prb, y_train, columns=[('prob'), ('y')])
df = pd.DataFrame({'prob':prb, 'y':y_train})
#df
df=df.sort_values('prob', ascending=False)
df = df.reset_index()
df['New_ID'] = df.index
df.head()
df['cum_y'] = df.y.cumsum()
df['cumy_perc'] = 100*df.cum_y/df.y.sum()
df['pop_perc'] = 100*df.New_ID/df.New_ID.max()
df['ks_value']=df['cumy_perc'] - df['pop_perc']
ksmax=df['ks_value'].max()
df.head(3)
print(ksmax)
max_pop =df.loc[df['ks_value'] == df['ks_value'].max(), 'pop_perc'].item()
print(max_pop)
df.tail(3)
plt.plot(df.pop_perc, df.cumy_perc, color='g')
plt.plot(df.pop_perc, df.pop_perc, color='orange')
plt.xlabel('Percent Population')
plt.ylabel('Percent Responded')
#plt.title('KS Lift Chart')
plt.text(40, 30, 'KSmax value is %d at %d percent' %(ksmax, max_pop))
plt.title('KS Lift Chart')
plt.show()
revenue=3200
cost=200
df['profit'] = df['cum_y']*3200 - df['New_ID']*200
max_pop =df.loc[df['profit'] == df['profit'].max(), 'pop_perc'].item()
prmax=df['profit'].max()
df.head(3)
print(max_pop)
df['profit'].max()
plt.plot(df.pop_perc, df.profit, color='g')
#plt.plot(df.pop_perc, df.pop_perc, color='orange')
plt.xlabel('Percent Population')
plt.ylabel('Profit')
#plt.title('KS Lift Chart')
plt.text(40, 30, 'Max profit is %d at %d percent' %(prmax, max_pop))
plt.title('Profit Curve')
plt.show()
dft = pd.read_excel("D:/REGI/1. Analytic Text Book/zSolutions/Chapter-8 Building Binary Models/Data/DSCH08LOGIRESW.xlsx", sheet_name="target")
dft.head(3)
dft.describe()
dft.info()
dft['card_tenure'].fillna(dft['card_tenure'].mean(), inplace=True)
dft['avg_bal'].fillna(dft['avg_bal'].mean(), inplace=True)
dft.describe()
dft['geo_group'].value_counts(dropna=False)
dft['res_type'].value_counts(dropna=False)
dft['geo_group'].fillna('E', inplace=True)
dft['res_type'].fillna('CO', inplace=True)
dft.info()
dfg=pd.get_dummies(dft['geo_group'], prefix='geo', drop_first=True)
dfr=pd.get_dummies(dft['res_type'], prefix='res', drop_first=True)
dfrg=dft.join([dfg,dfr])
dfrg.drop(['geo_group', 'res_type'], axis=1, inplace=True)
dft=dfrg
dft.head(3)

#X=dft.iloc[:, 1:]
Xt=dft.iloc[:, np.r_[1:3,4]]
Xt.head(3)
Xt.describe()
cid=dft['customer_id']
prb=logreg.predict_proba(Xt)[:,1]
tgt=pd.DataFrame({'resp_prob':prb, 'customer_id':cid })
tgt.sort_values('resp_prob', ascending=False, inplace=True)
tgt['tgt_num'] = range(len(tgt))
tgt['tgt_perc'] = 100*tgt.tgt_num/(tgt.tgt_num.max())
tgt.head(3)
target=tgt[tgt['tgt_perc']<=max_pop]
target.head()
target.info()
