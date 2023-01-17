# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dftrain = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
dftrain.head()
dftrain.shape
dftrain.isna().sum()
dftrain.TargetValue.value_counts()
from numba import vectorize, jit
import numpy as np
import pandas as pd
from time import time
import seaborn as sns

def test(a):
    return np.sqrt(a)

# numbatest = vectorize(['int64(int64)'], nopython=True, target="parallel")(test)
numbadefault = vectorize(nopython=True)(test)
numbaparallel = vectorize(['int64(int64)'], nopython=True, target="parallel")(test)
numpytest = np.vectorize(test, otypes = [np.uint64])
numbadefault(1)
numbaparallel(1)
numpytest(100)
import numba
numba.__version__
performance = pd.DataFrame()
for i in range(1000, 1000000, 50000): #, 
    print(f"Running {i} operations")
    arr = np.array(range(i))
    output = pd.DataFrame({
        'input' : arr
    })
    ### list comprehension
    start = time()
    output['ls'] = [test(a) for a in output['input']]
    end = time()
    lstime = end-start
    
    ## numba default
    start = time()
    output['ndefault'] = numbadefault(output['input'].values)
    end = time()
    ndtime = end-start

    ## numba parallel
    start = time()
    output['nparallel'] = numbaparallel(output['input'].values)
    end = time()
    numbaptime = end-start

    ### numpy
    start = time()
    output['numpy'] = numpytest(output['input'].values)
    end = time()
    nptime = end-start
 
    ### pandas eval
    start = time()
    output['pandas'] = output.eval('input**2')
    end = time()
    pdtime = end-start
    
#     ### pandas eval numba
#     start = time()
#     output['pandasnumba'] = output['input'].apply(lambda x: np.sqrt(x), engine='numba', raw=True)
#     end = time()
#     pdnumbatime = end-start
 
    
    obs = pd.DataFrame({
        'Elements' : [i]*5,
        'Type' : ['List Comprehension', 'Numba Vectorize', 'Numba Parallel Vectorize', 'Numpy Vectorize', 'Pandas eval'],
        'Time' : [lstime, ndtime, numbaptime, nptime, pdtime]
    })
    
    performance = pd.concat([performance, obs], axis=0)
sns.lineplot(x="Elements", y="Time", hue="Type", data=performance)
sns.lineplot(x="Elements", y="Time", hue="Type", data=performance[~performance.Type.isin(['List Comprehension', 'Numpy Vectorize'])])
### creating lags
def fn_lags_apply(df, colname, lags):
    """
    Function to create lags 
    """    
    if type(lags) == int:
        lagslist = list(range(1, lags + 1))
    else:
        lagslist = lags
    for x in lagslist:
        df.loc[:, colname + '_lag' + str(x)] = df['TargetValue'].shift(x).fillna(0)
    return df

def fn_lags_gsc(df, colname, lags):
    """
    Function to create lags 
    """    
    if type(lags) == int:
        lagslist = list(range(1, lags + 1))
    else:
        lagslist = lags
    
    for x in lagslist:        
        df.loc[:, colname + '_lag' + str(x)] = df.groupby(['Country_Region', 'Target']).shift(x)[colname].fillna(0)

    return df

def fn_lags_gcs(df, colname, lags):
    """
    Function to create lags 
    """    
    if type(lags) == int:
        lagslist = list(range(1, lags + 1))
    else:
        lagslist = lags
    
    for x in lagslist:        
        df.loc[:, colname + '_lag' + str(x)] = df.groupby(['Country_Region', 'Target'])[colname].shift(x).fillna(0)

    return df
performance = pd.DataFrame()
for i in range(1, 20):
    print(f"Adding {i} lags")
    starttime = time()
    dftrainwithlags = fn_lags_gcs(dftrain, 'TargetValue', i)
    endtime = time()
    gcs = endtime - starttime

    starttime = time()
    dftrainwithlags = fn_lags_gsc(dftrain, 'TargetValue', i)
    endtime = time()
    gsc = endtime - starttime

    starttime = time()
    dftrainwithlags = dftrain.groupby(['Country_Region', 'Target'], as_index=False).apply(fn_lags_apply, colname='TargetValue', lags=10)
    endtime = time()
    gapp = endtime - starttime

    
    
    obs = pd.DataFrame({
        'Lags' : [i]*3,
        'Type' : ['groupby()[colname].shift', 'groupby().shift()[colname]', 'groupby().apply'],
        'Time' : [gcs, gsc, gapp]
    })
    
    performance = pd.concat([performance, obs], axis=0)
sns.lineplot(x="Lags", y="Time", hue="Type", data=performance)
