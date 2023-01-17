# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
 
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA, ARMA
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('/kaggle/input/amazon-stock-price-20142019/AMZNtrain.csv')
print(data.head())
#We are only interested in the close price
df = data[['Date','Close']]
df.Date = pd.to_datetime(df.Date)
df = df.set_index("Date")
df.plot(style="-")
# Define the p, d and q parameters to take any value between 0 and 3
p = d = q = range(0, 3)
# Generate all different combinations of p, q and q
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore")
aic= []
parameters = []

for param in pdq:
    try:
        mod = sm.tsa.statespace.SARIMAX(df, order=param,
        enforce_stationarity=True, enforce_invertibility=True)
        results = mod.fit()
        # save results in lists
        aic.append(results.aic)
        parameters.append(param)
        #seasonal_param.append(param_seasonal)
        print('ARIMA{} - AIC:{}'.format(param, results.aic))
    except:
        continue
# find lowest aic          
index_min = min(range(len(aic)), key=aic.__getitem__)           
 
print('The optimal model is: ARIMA{} -AIC{}'.format(parameters[index_min], aic[index_min]))
model = ARIMA(df, order=parameters[index_min])
model_fit = model.fit(disp=0) 
print(model_fit.summary())
model_fit.plot_predict(start=2, end=len(df)+12)
plt.show()