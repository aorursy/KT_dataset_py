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
raw = pd.read_csv("/kaggle/input/hot-eda/hot_water_cleaned.csv", index_col = "timestamp", parse_dates = True)

raw.head(3)
raw = raw.resample("W").mean()

raw.head(3)
#creating dataframe for a single site



moose = pd.DataFrame()

M = [col for col in raw.columns if 'Moose' in col]

moose[M] = raw[M]



eagle = pd.DataFrame()

E = [col for col in raw.columns if 'Eagle' in col]

eagle[E] = raw[E]



cockatoo = pd.DataFrame()

C = [col for col in raw.columns if 'Cockatoo' in col]

cockatoo[C] = raw[C]



fox = pd.DataFrame()

f = [col for col in raw.columns if 'Fox' in col]

fox[f] = raw[f]



bobcat = pd.DataFrame()

bob = [col for col in raw.columns if 'Bobcat' in col]

bobcat[bob] = raw[bob]



crow = pd.DataFrame()

cr = [col for col in raw.columns if 'Crow' in col]

crow[cr] = raw[cr]



robin = pd.DataFrame()

R = [col for col in raw.columns if 'Robin' in col]

robin[R] = raw[R]



sites = [moose, eagle, cockatoo, robin, fox, bobcat, crow]
#adding a sum column

for site in sites:

    site["Hot_sum"] = site.sum(axis = 1)
bobcat.head(3)
from statsmodels.tsa.arima_model import ARIMA
model_eagle = eagle.copy()



train = model_eagle.iloc[0:(len(model_eagle)-30)]

test = model_eagle.iloc[len(train):(len(model_eagle)-1)]
#i dont know why the order in params is that, it just looks right?

endog = train["Hot_sum"]



mod = ARIMA(endog=endog, order=(2,0,1))

model_fit = mod.fit()

model_fit.summary()
from matplotlib import pyplot as plt



train['Hot_sum'].plot(figsize=(25,10))

model_fit.fittedvalues.plot()

plt.show()
predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1)

test['predicted'] = predict.values

test.tail(5)
test['predicted'].plot(color = 'red')

test["Hot_sum"].plot()