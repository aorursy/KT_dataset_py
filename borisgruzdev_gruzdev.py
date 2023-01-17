# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from multiprocessing.dummy import Pool as ThreadPool

from multiprocessing.dummy import Lock as ThreadLock

from multiprocessing.dummy import Value as ThreadValue

import functools

import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit

from tqdm.notebook import tqdm

from statsmodels.tsa.statespace.sarimax import SARIMAX

import statsmodels.api as sm

%matplotlib inline

from pylab import rcParams

import warnings

warnings.filterwarnings('ignore')

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from scipy.stats import linregress
def illustration(func):

    """

    Распаралеливание выкачки страниц.

    """

    mutex = ThreadLock()

    n_thread = ThreadValue('i',0)

    @functools.wraps(func)

    def wrapper(*args, **argv):

        result = func(*args, **argv)

        with mutex:

            nonlocal n_thread

            n_thread.value +=1

            if n_thread.value % 5 ==0:

                print(f"\r{n_thread.value} objects are processed...",end ='',flush = True)

        return result

    return wrapper
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, window = 5, cutoff = 0.05):



    #Determing rolling statistics

    rolmean = timeseries.rolling(window).mean()

    rolstd = timeseries.rolling(window).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 4))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    

    plt.show()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries.values,autolag='AIC' )

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    pvalue = dftest[1]

    if pvalue < cutoff:

        print('p-value = %.4f. The series is likely stationary.' % pvalue)

    else:

        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    

    return dfoutput
data_root = "/kaggle/input/wwwkagglecomborisgruzdev/"

df = pd.read_csv(data_root+'/train.csv',sep = ',')

df.sort_values(['sat_id','epoch'],axis = 0,inplace =True)

df['epoch'] = pd.to_datetime(df.epoch,format='%Y-%m-%d %H:%M:%S')

df.index  = df.epoch

df.drop('epoch', axis = 1, inplace = True)

df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

df.head()
df.info()
df_train = df[df.type == 'train']

df_test = df[df.type == 'test']
df_train['sat_id'].value_counts()
def plot_for_season(df,n,m,window = 10,y = 'error',x = 'epoch'):

    """

    Графики для времянных рядов различных спутников, необходимые для выявления 

    тренда.

    """

    fig,axes = plt.subplots(n,m,figsize = (20*(n/(n+m)),20*(m/(n+m))),

                            dpi = 120,sharey=False, sharex=False)

    n_groups = df_train.groupby(['sat_id']).ngroups

    for i in range(n):

        for j in range(m):

            #print("n == {},m == {}".format(n,m))

            indices = np.random.choice(n_groups,1)[0]

            axes[i][j].plot(df[df.sat_id == indices].index,

                            df[df.sat_id == indices][y])

            axes[i][j].plot(df[df.sat_id == indices].index

                            ,df[df.sat_id == indices][[y]].rolling(window = window).mean(),c = 'r')

            axes[i][j].set_title("for sat_id = {}".format(indices))

            axes[i][j].set_xlabel('{}'.format(x))

            axes[i][j].set_ylabel('{}'.format(y))

    fig.autofmt_xdate()

    fig.tight_layout()

    plt.show()
plot_for_season(df,4,4,24)# на множестве граффиков прекрасно видно, что сезонность равна 24

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#baseline

def predict_with_season_and_trend(train,test,period = 24):

    count = test.shape[0]//period

    res = test.shape[0]%period

    os1 = train.iloc[-period:].error.values.tolist()*count

    if res != 0:

        os1 += train.iloc[-period:-(period-res)].error.values.tolist()

    seas = np.array(os1)

    Mm = 48

    mm = 25

    trend = train.error[-Mm:].rolling(window=24).mean()

    s,*_  = linregress(np.arange(mm),trend.dropna().values)

    slope = s*np.arange(0,test.shape[0])

    seas+=slope

    return seas
@illustration

def predictions(indices):

    dff = df[df.sat_id == indices]

    df_traning = dff[dff.type == 'train']

    df_testing = dff[dff.type == 'test']

    pred = predict_with_season_and_trend(df_traning,df_testing,period = 24)

    a = [{'id' : df_testing["id"].values[i],

         'error_p' : pred[i]} for i in range(df_testing.shape[0])]

    return a
@illustration

def test_pred(indices):

    t_df = df_train[df_train.sat_id == indices]

    sh = np.int64(t_df.shape[0]*0.7)

    df_traning = t_df.iloc[:sh]

    df_testing = t_df.iloc[sh:]

    pred = predict_with_season_and_trend(df_traning,df_testing,period = 24)

    df_testing['pred_error'] = pred

    return mean_absolute_percentage_error(df_testing.error,df_testing.pred_error)
test_pred(0)
with ThreadPool(20) as pool:

    fold = pool.map(predictions,range(600))
fold = [x[i] for x in fold for i in range(len(x))]

kar = pd.DataFrame(fold)

kar
predictions = pd.merge(df[df.type == 'test'],kar,how = 'inner',on = 'id')

predictions.error = predictions.error_p

predictions.info()
new_df = pd.read_csv(data_root+'sub.csv',sep = ',')

our_preds = pd.merge(predictions,new_df,how = 'inner',on = 'id')

our_preds = our_preds[['id','error_x']]

our_preds.rename(columns={'error_x': 'error'}, inplace=True)

our_preds
our_preds.to_csv('/kaggle/output',sep=",",index = False)
our_preds.to_csv('/kaggle/working/pred.csv', index = False)