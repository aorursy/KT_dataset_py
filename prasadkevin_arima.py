# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import read_csv

import pandas as pd

from pandas import datetime

from pandas import DataFrame

from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

 

def parser(x):

	return datetime.strptime('190'+x, '%Y-%m')

 

series = pd.read_csv('../input/ARMA dataset.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# fit model

model = ARIMA(series, order=(5,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())

# plot residual errors

residuals = DataFrame(model_fit.resid)

residuals.plot()

pyplot.show()

residuals.plot(kind='kde')

pyplot.show()

print(residuals.describe())