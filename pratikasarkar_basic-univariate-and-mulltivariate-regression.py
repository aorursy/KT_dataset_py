import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.DataFrame({'bmi' : [4.0,5.5,6.8,7.2,7.8,9.8,9.7,8.8,11.0,13.0],

                'glucose' :[60,135,90,175,240,220,300,370,360,365]})
df
sns.scatterplot(df['bmi'],df['glucose'])

plt.plot([4,15],[300,550],color = 'r',linestyle = 'dashed')

plt.plot([4,16],[10,300],color = 'r',linestyle = 'dotted')
n_bmi = len(df['bmi'])

n_glucose = len(df['glucose'])
cov_bmi_glu = np.sum((df['bmi'] - np.mean(df['bmi']))*(df['glucose'] - np.mean(df['glucose'])))/(n_bmi-1)
var_bmi = np.sum((df['bmi'] - np.mean(df['bmi']))**2)/(n_glucose-1)
beta1 = cov_bmi_glu/var_bmi
beta1
ybar = np.mean(df['glucose'])

Xbar = np.mean(df['bmi'])
beta0 = ybar - beta1 * Xbar
beta0
glu_predict = beta0 + beta1 * df['bmi']
glu_predict
sns.scatterplot(df['bmi'],df['glucose'])

sns.lineplot(df['bmi'],glu_predict,color = 'g')
sse = np.sum((df['glucose'] - glu_predict)**2)

mse = sse/df.count()[0]

rmse = np.sqrt(mse)
rmse
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(df[['bmi']],df['glucose'])
lr.coef_
lr.intercept_
lr.predict(df[['bmi']])
r = np.corrcoef(df['glucose'],glu_predict)[0][1]

r
sns.scatterplot(df['glucose'],glu_predict)

plt.ylabel('predicted glucose values')

plt.xlabel('actual glucose values')
r_square = r**2

r_square
lr.score(df[['bmi']],df['glucose'])
import statsmodels.api as sm

from statsmodels.formula.api import ols
df_mtcars = pd.read_csv('../input/mtcars.csv')
df_mtcars.head()
df_mtcars['cyl'].value_counts()
df_mtcars.corr()
df_mtcars.columns
model = ols('mpg~cyl+disp+hp+drat+wt+qsec+vs+am+gear+carb',df_mtcars).fit()
model.params
model.summary()
model = ols('mpg~hp+wt',df_mtcars).fit()
model.params
model.summary()
mpg_pred = model.predict(df_mtcars[['hp','wt']])
rmse = np.sqrt(np.sum(((df_mtcars['mpg'] - mpg_pred)**2))/len(df_mtcars['mpg']))

rmse
import statsmodels.api as sm

sm.stats.diagnostic.linear_rainbow(model)
import statsmodels.stats.api as smi

smi.het_goldfeldquandt(model.resid,model.model.exog)
df_mtcars['cyl'].value_counts()
mpg_cyl_4 = df_mtcars[df_mtcars['cyl'] == 4]['mpg']

mpg_cyl_8 = df_mtcars[df_mtcars['cyl'] == 8]['mpg']

mpg_cyl_6 = df_mtcars[df_mtcars['cyl'] == 6]['mpg']
from scipy.stats import f_oneway

f_oneway(mpg_cyl_4,mpg_cyl_6,mpg_cyl_8)