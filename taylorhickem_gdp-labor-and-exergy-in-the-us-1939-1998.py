import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



ex_gdp_filename = '../input/gdp-exergy-ayres/US_GDP_exergy.csv'

emp_filename = '../input/gdp-exergy-ayres/bls_employment.csv'



#!ls ../input/gdp-exergy-ayres/



emp = pd.read_csv(emp_filename).dropna()[['Year','Period','Value']]

ex_gdp = pd.read_csv(ex_gdp_filename).dropna()

ex_gdp = ex_gdp[['year','GDP_bil','population_mil','exergy_cons','exergy_fin']]
ex_gdp.head()
emp.rename(columns={'Year':'year','Value':'emp_k'},inplace=True)

emp = emp[emp.Period=='M12'][['year','emp_k']]
emp.head()
ex_gdp = pd.merge(ex_gdp,emp,on='year')

ex_gdp['GDP_pc'] = ex_gdp['GDP_bil']/ex_gdp['emp_k']*10**6

ex_gdp['exg_pc'] = ex_gdp['exergy_cons']/ex_gdp['emp_k']*10**6
ex_gdp.tail()
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(ex_gdp[[x for x in ex_gdp if not x=='year']])

exgdp_norm = pd.DataFrame(x_scaled)

exgdp_norm.columns = [x for x in ex_gdp if not x=='year']

exgdp_norm.head()
fig, axes = plt.subplots(1,2,figsize=(12,6))

axes[0].plot('emp_k','GDP_bil',data=exgdp_norm)

axes[0].set_title('total emp v GDP')

axes[1].plot('exergy_cons','GDP_pc',data=exgdp_norm)

axes[1].set_title('total exergy v GDP per worker')
plt.plot('exg_pc','GDP_pc',data=exgdp_norm)

plt.title('exergy per capita v GDP per capita')
#add log of 3x main variables

exgdp_norm['log-GDP_pc'] = exgdp_norm.apply(lambda x:np.log(x['GDP_pc']),axis=1)

exgdp_norm['log-ex_pc'] = exgdp_norm.apply(lambda x:np.log(x['exg_pc']),axis=1)

exgdp_norm['log-ex'] = exgdp_norm.apply(lambda x:np.log(x['exergy_cons']),axis=1)
exgdp_norm[['exg_pc','exergy_cons','GDP_pc','log-GDP_pc']].corr()[['GDP_pc','log-GDP_pc']]
def get_model(emp_data,x_fields,y_field='GDP_pc'):

    x = emp_data[x_fields]

    y = emp_data[y_field]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state = 0)

    md = LinearRegression().fit(xTrain, yTrain)

    return (md,xTest,yTest)

(md,xTest,yTest) = get_model(exgdp_norm,['exergy_cons'])

#(md,xTest,yTest) = get_model(exgdp_norm,['emp_k','exergy_cons'])

ypred = md.predict(xTest)
r2_score(ypred.reshape(-1,1),pd.DataFrame(yTest).as_matrix())
md.coef_
plt.scatter(ypred,yTest, color = 'red')

plt.plot(yTest, yTest, color = 'blue')

plt.title('GDP prediction accuracy')

plt.xlabel('GDP actual')

plt.ylabel('GDP linear model - population, exergy')

plt.show()
ypred_all = md.predict(pd.DataFrame(exgdp_norm['exergy_cons']))
plt.scatter(ex_gdp['year'],exgdp_norm['GDP_pc'],label='GDP per capita actual',color = 'red')

plt.plot(ex_gdp['year'], ypred_all,label='GDP = f(exergy)', color = 'blue')

plt.plot(ex_gdp['year'], exgdp_norm['exg_pc'],label='exergy pc', color = 'green')

plt.title('GDP prediction over time')

plt.xlabel('year')

plt.legend()

#plt.ylabel('GDP linear model - population, exergy')

plt.show()