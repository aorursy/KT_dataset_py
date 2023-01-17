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
import pandas as pd 
from statsmodels.tsa.statespace.sarimax import SARIMAX
%matplotlib inline
df= pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
df.head()
df.index=df.Date
df.plot()
sarimax_model=SARIMAX(df['Price'],Order=(1,1,1),seasonal_order=(1,1,1,4))
res=sarimax_model.fit(disp=False)
res.summary()