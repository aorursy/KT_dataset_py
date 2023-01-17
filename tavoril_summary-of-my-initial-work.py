import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
%matplotlib inline
data = pd.read_csv('../input/credit_card_clients_split.csv')
train = data.dropna().copy()
train['log_income']=np.log10(1 + train.income)
train['log_turnover']=np.log10(1 + train.avg_monthly_turnover)
train['positive_turnover']=(train.log_turnover!=0).astype(int)
train[train.positive_turnover == 1].age.hist(density=True, range=[20,65], bins=25)
train[train.positive_turnover == 0].age.hist(density=True, range=[20,65], alpha=0.5, bins=25)
plt.legend(['positive turnover', 'zero turnover'])
plt.title('Histogram of age conditional on zero or positive turnover');
train.groupby('positive_turnover').log_turnover.count()/len(train)
(train.groupby(['age','positive_turnover']).size()/train.groupby('age').size()).unstack()
train['age_sq']=train.age**2
train['young']=(train.age<=28).astype(int)
train=pd.concat([train,pd.get_dummies(train.wrk_rgn_code, prefix="rgn")], axis=1)
train=pd.concat([train,pd.get_dummies(train.education)], axis=1)
train=pd.concat([train,pd.get_dummies(train.sales_channel_id, prefix="sales")], axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train.columns
columns_to_keep=['age']+train.loc[:,"log_income":].columns.tolist()
columns_to_keep=columns_to_keep[:2]+columns_to_keep[4:]
columns_to_predict=['positive_turnover','log_turnover']
X=train[columns_to_keep]
y=train[columns_to_predict]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(X_train, y_train.positive_turnover)  
logreg.fit(X_train.loc[:,rfe.support_],y_train.positive_turnover)
y_pred=logreg.predict(X_test.loc[:,rfe.support_])
accuracy_score(y_test.positive_turnover,y_pred)
X_test['positive_pred']=logreg.predict(X_test.loc[:,rfe.support_])
X_test.groupby('positive_pred').size()
X_test.drop('positive_pred',axis=1,inplace=True)
p=pd.DataFrame(data=logreg.predict_proba(X_test.loc[:,rfe.support_]), columns=['prob_zero', 'prob_positive'])
p.set_index(X_test.index, inplace=True)
X_test=X_test.join(p)
X_test.head()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test, y_test, test_size=0.33, random_state=100)
def backwardElimination(X, Y):
    for i in np.arange(0.9,0,-0.1):
        numVars = len(X.columns.tolist())
        d=[]
        regressor_OLS = sm.OLS(Y, X).fit()
        for j in range(numVars):
            if regressor_OLS.pvalues[j] > i:
                d.append(X.columns[j])
        for t in d:
            X=X.drop(t, axis=1)
    
    regressor_OLS = sm.OLS(Y, X).fit()
    while max(regressor_OLS.pvalues > i):
        numVars = len(X.columns.tolist())
        d=[]
        for j in range(numVars):
            if regressor_OLS.pvalues[j] > i:
                d.append(X.columns[j])
        for t in d:
            X=X.drop(t, axis=1)
            
        regressor_OLS = sm.OLS(Y, X).fit()
    print(regressor_OLS.summary())
    #predictions = regressor_OLS.predict(X)
    return regressor_OLS, X.columns.tolist()
regressor, ctp=backwardElimination(X_train2, y_train2.log_turnover)
X_test2['mean_log_turnover']=regressor.predict(X_test2[ctp])
X_test2.head()
rows_to_check=X_test2.index.tolist()
train_tested=train.loc[rows_to_check,:]
train_tested.shape
np.mean((X_test2.mean_log_turnover- np.log10(1+train_tested['avg_monthly_turnover']))**2)
data['age_sq']=data.age**2
data['young']=(data.age<=28).astype(int)
data['log_income']=np.log10(1 + data.income)
data['log_turnover']=np.log10(1 + data.avg_monthly_turnover)
data=pd.concat([data,pd.get_dummies(data.wrk_rgn_code, prefix="rgn")], axis=1)
data=pd.concat([data,pd.get_dummies(data.education)], axis=1)
data=pd.concat([data,pd.get_dummies(data.sales_channel_id, prefix="sales")], axis=1)
data.head()
X2_data=data[columns_to_keep]
p_data=pd.DataFrame(data=logreg.predict_proba(X2_data.loc[:,rfe.support_]), columns=['prob_zero', 'prob_positive'])
p_data.set_index(data.index, inplace=True)
data=data.join(p_data)
print(data.shape)
data.head()
data['mean_log_turnover']=regressor.predict(data[ctp])
data['predicted_turnover'] = 10 ** data['mean_log_turnover']
submission = data.loc[data.avg_monthly_turnover.isnull(), ['id', 'predicted_turnover']].copy()
submission.columns = ['id', 'avg_monthly_turnover']
submission.to_csv('kernel_to_upload.csv', index=None)
