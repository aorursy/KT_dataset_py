# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import *

# import sklearn.model_selection as skm

# from sklearn.linear_model import LogisticRegressionCV

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filn = "../input/"+'HR_comma_sep.csv'

df = pd.read_csv(filn)

print(len(df))
df.head()
from matplotlib.pyplot import *

import seaborn as sns



satlev  = df['satisfaction_level']

atbins = 50

fig, ax = subplots(1,1,figsize=(10,6))

sns.distplot(satlev, kde=True, color="b", bins=atbins)

title('Histogram of Satisfaction Level @'+str(atbins)+' bins')
df.describe()
from collections import Counter

prom5 = df['promotion_last_5years']

print(Counter(prom5))
print(Counter(df['left']))
print(float(3571)/11428)
print(Counter(df['sales']))

print('\n',Counter(df['salary']))
fig3, ax3=subplots(1,1,figsize=(12,6))

df['sales'].value_counts().plot(kind='bar',ax=ax3,color='r')

title('Sales Distribution')

fig4, ax4 = subplots(1,1,figsize=(8,8))

df['salary'].value_counts().plot(kind='pie',ax=ax4,autopct='%.3f')

title('Salary Distribution')
from statsmodels.tools import categorical

df0 = df.copy()

df_dict = {'low':0, 'medium':1, 'high':2}

        

df0['salnum'] = df['salary'].map(df_dict)

df0.drop(['salary'],axis=1,inplace=True)

df0.head(3)
sss = df0[['satisfaction_level','sales','salnum']]

# sss1 = pd.melt(sss, "sales", var_name="meas")

figss, axss = subplots(1,1,figsize=(14,8))

sns.kdeplot(sss.satisfaction_level, sss.salnum, 

            palette="dark", shade=True, shade_lowest=False, hue='sales')

title('Density plot of Salnum and Satisfaction Level')


figsm, axsm = subplots(1,1,figsize=(14,8))

sns.violinplot(x="sales", y="satisfaction_level", hue="salary", data=df, palette='pastel')
import statsmodels.api as sma

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegressionCV
x = df.drop(['left','sales','salary'],axis=1)

x['salnum'] = sss['salnum']

y = df['left']
xtrn, xtst, ytrn, ytst = train_test_split(x,y)

lm = LogisticRegressionCV(fit_intercept=True, cv=3)

lmfit = lm.fit(xtrn,ytrn)
ppred = lmfit.predict_proba(xtst)

pred = lmfit.predict(xtst)



from sklearn.metrics import accuracy_score, log_loss, precision_score

pscore = log_loss(ytst, ppred)

score = accuracy_score(ytst,pred)

print("Accuracy = ", score, "\n")

print("Log Prob = ", pscore)
mod = sma.Logit(endog=y,exog=sma.add_constant(x))

fit = mod.fit()

print(fit.summary())
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', 

                           algorithm='auto', leaf_size=30, 

                           p=2, metric='minkowski', 

                           metric_params=None, n_jobs=-1)

knnfit = knn.fit(xtrn,ytrn)

knnpred = knnfit.predict(xtst)

score = accuracy_score(ytst,knnpred)

prec = precision_score(ytst,knnpred)

print("Accuracy = %.3f" % score,'%', " \n")

print("Precision = %.3f" % prec,'%', " \n")
x2 = df.drop(['sales','salary'],axis=1)

x2['salnum'] = x['salnum']

figh, axh=subplots(1,1,figsize=(10,8))

sns.heatmap(x2.corr(),ax=axh)
x3 = df[['last_evaluation','number_project','average_montly_hours','time_spend_company']]

y3 = df[['satisfaction_level']]
from sklearn.svm import SVR, LinearSVR
xtrn3, xtst3, ytrn3, ytst3 = train_test_split(x3,y3)

svReg = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',

    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

## USing just the defaults

svrFit = svReg.fit(xtrn3,np.array(ytrn3).ravel())
svrPreds = svrFit.predict(xtst3)

svrP = pd.DataFrame(svrPreds, index=ytst3.index, columns=['SVR RBF Predictions'])
svrL = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', 

                 fit_intercept=True, intercept_scaling=1.0, dual=True, 

                 verbose=0, random_state=None, max_iter=1000)

svrLFit = svrL.fit(xtrn3,np.array(ytrn3).ravel())
svrPL = pd.DataFrame( (svrLFit.predict(xtst3)), index=ytst3.index, columns=['Linear SVR Preds'])
from sklearn.metrics import mean_squared_error, r2_score

rbfmse = mean_squared_error(ytst3,svrP)

linmse = mean_squared_error(ytst3,svrPL)

rbfR2 = r2_score(ytst3,svrP)

linR2 = r2_score(ytst3,svrPL)

print("Mean Squared Error SVR-RBF = %0.4f" % rbfmse, '  RSqd = %0.2f' % rbfR2, "\n")

print("Mean Squared Error SVR-Lin = %0.4f" % linmse, '  RSqd = %0.2f' % linR2)
sm2 = sma.OLS(ytrn3,xtrn3)

sm2f = sm2.fit()

sm2p = pd.DataFrame(sm2f.predict(xtst3), index=ytst3.index, columns=['StatMod OLS Pred'])

print(sm2f.summary2())
fig4, ax4=subplots(1,1,figsize=(12,8))

ytst3.plot(ax=ax4, style=['r.'], legend=True)

svrP.plot(style=['b.'],ax=ax4,legend=True)

svrPL.plot(style=['g.'],ax=ax4,legend=True)

sm2p.plot(ax=ax4, style=['c.'], legend=True)

legend(loc='best')

title('Various Regression Methods')
full = pd.concat([ytst3,svrP,svrPL,sm2p],axis=1,join='outer')

full.columns = ['ACT','RBF','LIN','OLS']
fig5, ax5 = subplots(1,4,sharey=True,figsize=(12,12))

ax5 = ax5.ravel()

sns.kdeplot(full.ACT, full.RBF, cmap="Reds", shade=True, shade_lowest=False, ax=ax5[0])

sns.kdeplot(full.ACT, full.LIN, cmap="Blues", shade=True, shade_lowest=False, ax=ax5[1])

sns.kdeplot(full.ACT, full.OLS, cmap="Greens", shade=True, shade_lowest=False, ax=ax5[2])

sns.kdeplot(full.ACT, full.ACT, cmap="Purples", shade=True, shade_lowest=False, ax=ax5[3])

title('Kernel Desnsities of Predictions')