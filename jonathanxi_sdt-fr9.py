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
from matplotlib import pyplot as plt

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import warnings

warnings.filterwarnings('ignore')
!pip install pmdarima

import pmdarima as pm
import seaborn as sns

sns.set(rc={"figure.figsize": (8, 6)}); 

sns.set(style="white")
path = '../input/sdtdataset/'

df_company=pd.read_csv(f'{path}Company.csv')

df_real=pd.read_csv(f'{path}Real Property.csv')

df_company.tail()
df_real.tail()
def data_plot(df):

    df.index=df['year']

    df['index_log']=df['index'].apply(np.log)

    fig = plt.figure(figsize=(16,12))

    ax1= fig.add_subplot(2,2,1)

    ax2= fig.add_subplot(2,2,2)

    df['index'].plot(ax=ax1,title='Index',style=['+-'])

    df['index_log'].plot(ax=ax2,title='Log Index',style=['d--'])

    sns.despine()
data_plot(df_company)
def get_stationarity(df):

    

    # Dickey–Fuller test:

    result = adfuller(df)

    print('ADF Statistic: {}'.format(result[0]))

    print('p-value: {}'.format(result[1]))

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t{}: {}'.format(key, value))
def transformation(df):

    rolling_mean_exp_decay = df['index_log'].ewm(halflife=26, min_periods=0, adjust=False).mean()

    df_log_exp_decay = df['index_log'] - rolling_mean_exp_decay

    df_log_exp_decay.dropna(inplace=True)

    df['ewm']= df_log_exp_decay

    df['rolling_mean']=rolling_mean_exp_decay

    get_stationarity(df_log_exp_decay)
get_stationarity(df_company['index'])
transformation(df_company)
pm.auto_arima(df_company['ewm'], trace=True, error_action='ignore', suppress_warnings=True,seasonal=False)
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_company['ewm'], order=(2,0,0))

results=model.fit(disp=-1)

results.plot_predict(1,len(df_company)+16)

sns.despine()
def test(results):

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plot_acf(results.resid.values)

    plt.title('ACF for original dataset',fontsize=10, fontweight="bold")

    sns.despine()

    import statsmodels.api as sm

    from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test

    print('Box Test and ACF:')

    return sm.stats.acorr_ljungbox(results.resid, lags=[10])

    
test(results)
def prediction_1(df,a):

    model = ARIMA(df['rolling_mean'], order=(1,0,0))

    results_rolling = model.fit(disp=-1)

    rolling_Log_exp=results_rolling.predict(len(df_company),len(df_company)+16)

    

    ##这里是之前那个模型得结果：

    prediction_values=results.predict(len(df_company),len(df_company)+16)

    fc, se, conf = results.forecast(17, alpha=a)  #这里可以用来设置置信度

    interval_bottom=conf[:,0]

    interval_up=conf[:,1]



    prediction_result=pd.DataFrame(np.exp(interval_bottom+rolling_Log_exp))

    prediction_result['1']=np.exp(prediction_values+rolling_Log_exp)

    prediction_result['2']=np.exp(interval_up+rolling_Log_exp)

    prediction_result.columns=['Bottom','Fitted','Up']

    

    return prediction_result
prediction_1(df_company,0.05)
pm.auto_arima(df_company['index'], trace=True, error_action='ignore', suppress_warnings=True,seasonal=False,information_criterion='aic',intercept=True,stationary=True)
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_company['index'], order=(2,0,0))

results=model.fit(disp=-1)

results.plot_predict(1,len(df_company)+16)

sns.despine()
test(results)
def prediction_2(a):

    

    ##这里是上面那个模型得结果：

    prediction_values=results.predict(len(df_company),len(df_company)+16)

    fc, se, conf = results.forecast(17, alpha=a)  #这里可以用来设置置信度

    interval_bottom=conf[:,0]

    interval_up=conf[:,1]



    prediction_result=pd.DataFrame(interval_bottom)

    prediction_result['1']=prediction_values.values

    prediction_result['2']=interval_up

    prediction_result.columns=['Bottom','Fitted','Up']

    

    return prediction_result
prediction_2(0.05)