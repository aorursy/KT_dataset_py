# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv("../input/train_1.csv")

test = pd.read_csv("../input/key_1.csv")
#zero_serie = 0*np.ones(test.shape[0])

#dfTest = pd.DataFrame({'Id': test.Id.values, 'Visits': zero_serie})



#dfTest[['Id','Visits']].to_csv('first_submit.csv', index=False)
#dfTest2 = test.copy()

#dfTrain2 = train.copy()

#dfTest2['Page'] = dfTest2.Page.apply(lambda a: a[:-11])

#dfTrain2['Visits'] = dfTrain2.drop('Page', axis=1).mean(axis=1, skipna=True)

#dfTest2 = dfTest2.merge(dfTrain2[['Page','Visits']], how='left')

#dfTest2.loc[dfTest2.Visits.isnull(), 'Visits'] = 0

#dfTest2.drop('Page', axis=1)

#dfTest2[['Id','Visits']].to_csv('mean_submit.csv', index=False)
#dfTest3 = test.copy()

#dfTrain3 = train.copy()

#dfTest3['Page'] = dfTest3.Page.apply(lambda a: a[:-11])

#dfTrain3['Visits'] = dfTrain3.drop('Page', axis=1).median(axis=1, skipna=True)

#dfTest3 = dfTest3.merge(dfTrain3[['Page','Visits']], how='left')

#dfTest3.loc[dfTest3.Visits.isnull(), 'Visits'] = 0

#dfTest3.drop('Page', axis=1)

#dfTest3[['Id','Visits']].to_csv('median_submit.csv', index=False)
from fbprophet import Prophet

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
from numba import jit

import math



@jit

def smape_fast(y_true, y_pred):

    out = 0

    for i in range(y_true.shape[0]):

        a = y_true[i]

        b = y_pred[i]

        c = a+b

        if c == 0:

            continue

        out += math.fabs(a - b) / c

    out *= (200.0 / y_true.shape[0])

    return out
def runProphet(_id, _start, _frontier, _data):

    _data=_data.iloc[_id,:]

    data_train, data_test = _data.iloc[_start:-_frontier], _data.iloc[-_frontier:]

    test_median = data_test.median()

    test_cleaned = data_test.T.fillna(test_median).T

    train_median = data_train.iloc[1:].median()

    train_cleaned = data_train.T.iloc[1:].fillna(train_median).T

    data=train_cleaned.iloc[:].to_frame()

    data.columns = ['visits']

    #fill outliers that are out of 1.5*std with rolling median of 56 days

    data['median'] = data.visits.median()

    std_mult = 1.0

    data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'visits'] = data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'median']

    data.index = pd.to_datetime(data.index)

    #prophet expects the folllwing label names

    X = pd.DataFrame(index=range(0,len(data)))

    X['ds'] = data.index

    X['y'] = data['visits'].values  

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)

    m.fit(X)

    future = m.make_future_dataframe(periods=_frontier)

    forecast = m.predict(future)

    m.plot(forecast);

    y_truth = test_cleaned

    y_forecasted = forecast.iloc[-_frontier:,2].values

    score = smape_fast(y_truth, y_forecasted)

    print(score)
runProphet(5, 0, 60, train)