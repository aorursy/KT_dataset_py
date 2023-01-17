import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso
os.chdir('/kaggle/input')

wvs = pd.read_csv("wvs.csv.bz2",sep="\t")

wvs.sample(7)
wvs = wvs.dropna(subset=['V23'])

#removing all observations with negative responses

wvs = wvs[(wvs.V23 > 0)]

wvs.shape
#No of people voting for each satisfaction level

wvs.groupby('V23').size()
#Proportion of people that are 6 or more satisfied

wvs[(wvs.V23 >= 6)].shape[0]/wvs.V23.shape[0]*100
#Histogram of poeple voting for each level of life satisfaction

plt.hist(wvs.V23, alpha=0.6)

plt.axvline(wvs.V23.mean(), color='k', linestyle='dashed', linewidth=1)



min_ylim, max_ylim = plt.ylim()

plt.text(wvs.V23.mean()*0.7, max_ylim*0.9, 'Mean: {:.2f}'.format(wvs.V23.mean()))
#Variable selection

num_columns = ['V23','V4','V5','V6','V7','V8','V9','V10','V11','V24','V45','V47','V48','V49','V50','V51','V52','V53','V54',

               'V55','V56','V59','V70','V71','V72','V73','V74','V75','V76','V77','V78','V79','V84','V102','V103','V104',

               'V105','V106','V107','V108','V109','V110','V111','V113','V114','V115','V116','V117','V118','V119','V120',

               'V121','V122','V124','V127','V128','V130','V131','V132','V133','V134','V135','V136','V137','V138','V139',

               'V140','V141','V142','V143','V145','V146','V152','V160','V164','V165','V166','V167','V170','V171','V172',

               'V181','V182','V183','V184','V185','V186','V188','V189','V190','V191','V192','V193','V194','V195','V196',

               'V197','V198','V199','V200','V201','V202','V203','V204','V205','V207','V208','V209','V210','V242']



cat_columns = {'V2':'country','V240':'sex','V250':'live_w_parents','V57':'relationship','V82':'donate','V83':'env_cause',

              'V151':'life_sense','V179':'crime','V80':'problem','V147':'religiousness'}



all_columns = num_columns.copy()

all_columns.extend(['V2','V240','V250','V57','V82','V83','V151','V179','V80','V147'])
#Data cleaning

wvs_select = wvs[all_columns]

print("Shape of selected:",wvs_select.shape)

wvs_select = wvs_select.dropna()

print("Shape after dropping NA:",wvs_select.shape)

wvs_pos = wvs_select[(wvs_select > 0).all(1)]

print("Shape after removing all negative values",wvs_pos.shape)
for k,v in cat_columns.items():

    print(k,v,)

    wvs_pos = wvs_pos.rename(columns={k:v})

    wvs_pos = pd.get_dummies(wvs_pos,columns=[v],drop_first=True)

    print("After adding",v,"number of variables is:",wvs_pos.shape[1],"condition number k=",np.linalg.cond(wvs_pos))

print("Final shape of dataset is",wvs_pos.shape)
y = wvs_pos['V23']

X = wvs_pos.drop(['V23'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
ridge = Ridge(alpha=0.1).fit(X_train,y_train)

y_pred = ridge.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
ridge = Ridge(alpha=1e-8).fit(X_train,y_train)

y_pred = ridge.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
ridge = Ridge(alpha=10).fit(X_train,y_train)

y_pred = ridge.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
ridge = Ridge(alpha=1e5).fit(X_train,y_train)

y_pred = ridge.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
lasso = Lasso(alpha=0.1).fit(X_train,y_train)

y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
lasso = Lasso(alpha=1e-8).fit(X_train,y_train)

y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
lasso = Lasso(alpha=10).fit(X_train,y_train)

y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
lasso = Lasso(alpha=1).fit(X_train,y_train)

y_pred = lasso.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
def vary_lambda(X,y,Xt,yt):

    rmse_col = ['lambda','log_l','ridge','lasso']

    rmse = pd.DataFrame(columns=rmse_col)

    coef_col = ['lambda','log_l','ridge0','ridge1','ridge2','lasso0','lasso1','lasso2']

    coef = pd.DataFrame(columns=coef_col)

    

    #linear model

    m = LinearRegression().fit(X, y)

    yhat = m.predict(Xt)

    lm_rmse = np.sqrt(np.mean((yt - yhat)**2))

    lm_coef = m.coef_

    

    for l in [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000,1e6,1e7,1e8]:

        # penalty value

        alpha=l

        mr = Ridge(alpha=alpha).fit(X, y)

        yhatr = mr.predict(Xt)

        rmser = np.sqrt(np.mean((yt - yhatr)**2))

        coefr = mr.coef_

        # large alpha = large penalty

        ml = Lasso(alpha=alpha).fit(X, y)

        yhatl = ml.predict(Xt)

        rmsel = np.sqrt(np.mean((yt - yhatl)**2))

        coefl = ml.coef_

        #storing only first three coefficients of Lasso and Ridge for simpler comparison

        coef.loc[len(coef)] = [l,np.log(l),coefr[0],coefr[1],coefr[2],coefl[0],coefl[1],coefl[2]]

        rmse.loc[len(rmse)] = [l,np.log(l),rmser,rmsel]

    coef['lm0'] = lm_coef[0]

    coef['lm1'] = lm_coef[1]

    coef['lm2'] = lm_coef[2]

    rmse['lm'] = lm_rmse

    return (rmse,coef)
rmse,coef = vary_lambda(X_train,y_train,X_test,y_test)
rmse
plt.figure(figsize=(7,7))

plt.plot('log_l', 'ridge', data=rmse, label='Ridge')

plt.plot('log_l', 'lasso', data=rmse, label='Lasso')

plt.plot('log_l', 'lm', data=rmse, label='Linear')

plt.xlabel('Log Alpha')

plt.ylabel('RMSE')

plt.title('RMSE for Linear, Ridge and Lasso regression')

plt.legend()
plt.figure(figsize=(6,6))

plt.plot('log_l', 'ridge0', data=coef, label='Ridge coef0', color='#00FFFF')

plt.plot('log_l', 'lasso0', data=coef, label='Lasso coef0', color='#ff0000')

plt.plot('log_l', 'lm0', data=coef, label='LM coef0', color='#dcdcdc')

plt.xlabel('Log Alpha')

plt.ylabel('First Coefficients')

plt.title('Coefficient0 for Linear, Ridge and Lasso Models')

plt.legend()
plt.figure(figsize=(6,6))

plt.plot('log_l', 'ridge1', data=coef, label='Ridge coef1', color='#000080')

plt.plot('log_l', 'lasso1', data=coef, label='Lasso coef1', color='#b20000')

plt.plot('log_l', 'lm1', data=coef, label='LM coef1', color='#c0c0c0')

plt.xlabel('Log Alpha')

plt.ylabel('Coefficients')

plt.title('Coefficient1 for Linear, Ridge and Lasso Models')

plt.legend()
plt.figure(figsize=(6, 6))

plt.plot('log_l', 'ridge2', data=coef, label='Ridge coef2', color='#0000FF')

plt.plot('log_l', 'lasso2', data=coef, label='Lasso coef2', color='#7f0000')

plt.plot('log_l', 'lm2', data=coef, label='LM coef2', color='#808080')

plt.xlabel('Log Alpha')

plt.ylabel('Coefficients')

plt.title('Coefficient2 for Linear, Ridge and Lasso Models')

plt.legend()
plt.figure(figsize=(7,7))

plt.plot('log_l', 'ridge0', data=coef, label='Ridge coef0', color='#00FFFF')

plt.plot('log_l', 'lasso0', data=coef, label='Lasso coef0', color='#ff0000')

plt.plot('log_l', 'ridge1', data=coef, label='Ridge coef1', color='#000080')

plt.plot('log_l', 'lasso1', data=coef, label='Lasso coef1', color='#b20000')

plt.plot('log_l', 'ridge2', data=coef, label='Ridge coef2', color='#0000FF')

plt.plot('log_l', 'lasso2', data=coef, label='Lasso coef2', color='#7f0000')

plt.xlabel('Log Alpha')

plt.ylabel('First Coefficients')

plt.title('Coefficients for Ridge and Lasso Models')

plt.legend()
X_3 = X[['V11','V55','V59']]

X_3_y  = X_3.copy()

X_3_y['satisfaction'] = y
m = smf.ols(formula='satisfaction ~ V11 + V55 + V59', data=X_3_y).fit()

print(m.summary())
pred_y = m.predict(X_3)

np.sqrt(mean_squared_error(y, pred_y))