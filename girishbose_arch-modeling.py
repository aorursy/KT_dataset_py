# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install arch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.simplefilter('ignore')

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style('darkgrid')
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font",family="sans-serif")
plt.rc("font",size=14)
from arch import arch_model
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv',index_col="Date",parse_dates=True)
crd_inflation = 100 * data.pct_change().dropna()
fig = crd_inflation.plot()
from arch import arch_model

am = arch_model(crd_inflation)
res = am.fit(update_freq=5)
print(res.summary())
am = arch_model(crd_inflation,p=1,o=1,q=1,power=1.0)
res = am.fit(update_freq=5)
print(res.summary)
fig = res.plot(annualize = 'M')
am = arch_model(crd_inflation,p=1,o=0,q=1,power=1.0,dist='StudentsT')
res = am.fit(update_freq=5)
print(res.summary())
from collections import OrderedDict
crude_ret = 100 * data.dropna().pct_change().dropna()
res_normal = arch_model(crude_ret).fit(disp='off')
res_t = arch_model(crude_ret, dist='t').fit(disp='off')
res_skewt = arch_model(crude_ret, dist='skewt').fit(disp='off')
lls = pd.Series(
    OrderedDict((('normal', res_normal.loglikelihood),
                 ('t', res_t.loglikelihood), ('skewt',
                                              res_skewt.loglikelihood))))
print(lls)
params = pd.DataFrame(
    OrderedDict((('normal', res_normal.params), ('t', res_t.params),
                 ('skewt', res_skewt.params))))
params
std_resid = res_normal.resid / res_normal.conditional_volatility
unit_var_resid = res_normal.resid / res_normal.resid.std()
df = pd.concat([std_resid, unit_var_resid], 1)
df.columns = ['Std Resids', 'Unit Variance Resids']
subplot = df.plot(kind='kde', xlim=(-4, 4))
res = arch_model(crude_ret, p=1, o=1, q=1, dist='skewt').fit(disp='off')
pd.DataFrame(res.params)
sim_mod = arch_model(None, p=1, o=1, q=1, dist="skewt")

sim_data = sim_mod.simulate(res.params, 1000)
sim_data.head()
from arch.univariate import ConstantMean, GARCH, SkewStudent
import numpy as np

rs = np.random.RandomState([892380934, 189201902, 129129894, 9890437])
# Save the initial state to reset later
state = rs.get_state()

dist = SkewStudent(random_state=rs)
vol = GARCH(p=1, o=1, q=1)
repro_mod = ConstantMean(None, volatility=vol, distribution=dist)

repro_mod.simulate(res.params, 1000).head()
# Reset the state to the initial state
rs.set_state(state)
repro_mod.simulate(res.params, 1000).head()