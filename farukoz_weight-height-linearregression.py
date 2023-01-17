# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

5# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from warnings import filterwarnings
filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/heights-and-weights/data.csv')
data
data.info()
data.describe()
data.isnull().values.any()
data.corr()
import seaborn as sns
sns.pairplot(data, kind = 'reg' );
sns.jointplot(x = 'Weight', y = 'Height', data = data , kind = 'reg' )
import statsmodels.api as sm
X = data[['Height']]
X[0:5]
X = sm.add_constant(X) # iki boyutlu yapıda olması için
X[0:5]
y = data['Weight']
y[0:5]
lm = sm.OLS(y,X)
model = lm.fit()
model.summary()
model.params
model.mse_model
print('Weight = ' + str('%.2f ' % model.params[0]) +  '+ Height * ' + str('%.2f' % model.params[1]))
g = sns.regplot(data['Height'], data['Weight'], ci = None, scatter_kws = {'color':'r', 's':50})
g.set_title('Model Denklemi:Weight = -39.06 + Height * 61.27')
g.set_ylabel('Weight')
g.set_xlabel('Height')
import matplotlib.pyplot as plt
plt.xlim(1.4,1.9)
plt.ylim(45,80)
from sklearn.linear_model import LinearRegression
X = data[['Height']]
y = data['Weight']
reg = LinearRegression()
model = reg.fit(X,y)
model.intercept_,model.coef_
model.score(X,y)
model.predict(X)[0:10]
veri = [[0],[1],[2]]
model.predict(veri)
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
lm = smf.ols('Weight ~ Height', data)
model = lm.fit()
mse = mean_squared_error(y, model.fittedvalues)
mse
import numpy as np
rmse = np.sqrt(mse)
rmse
reg.predict(X)[0:10] # tahmini değerler
y[0:10] # gerçek değerler
k_t = pd.DataFrame({'gerçek_y': y[0:10],
                   'tahmini_y':reg.predict(X)[0:10]})
k_t
k_t['Hata'] = k_t['gerçek_y']-k_t['tahmini_y']
k_t
k_t['Hata_kare'] = k_t['Hata']**2
k_t
np.sum(k_t['Hata_kare'])
np.mean(k_t['Hata_kare'])
np.sqrt(np.mean(k_t['Hata_kare']))
model.resid[0:10]
plt.plot(model.resid)