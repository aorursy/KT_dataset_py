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
import pandas as pd

df=pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_levels.csv",parse_dates=["Date"],index_col="Date")

df.head()
df["2019-04"]
df["2019-04"]["REDHILLS"]
df["2019-04"].REDHILLS.mean()
df1=df.REDHILLS.resample('M').mean()

df1
%matplotlib inline

df.REDHILLS.resample('M').mean().plot()
%matplotlib inline

df.CHOLAVARAM.resample('M').mean().plot()
%matplotlib inline

df.POONDI.resample('M').mean().plot()
%matplotlib inline

df.CHEMBARAMBAKKAM.resample('M').mean().plot()
df.CHEMBARAMBAKKAM.resample('Q').mean().plot()
df.POONDI.resample('Y').mean().plot()
df.CHEMBARAMBAKKAM.resample('Y').mean().plot()
df.CHOLAVARAM.resample('Y').mean().plot()
df.REDHILLS.resample('Y').mean().plot()
import pandas as pd

df11=pd.read_csv("/kaggle/input/chennai-water-management/chennai_reservoir_rainfall.csv",parse_dates=["Date"],index_col="Date")



df11
%matplotlib inline

df11.REDHILLS.resample('M').mean().plot()
%matplotlib inline

df11.POONDI.resample('M').mean().plot()
%matplotlib inline

df11.CHOLAVARAM.resample('M').mean().plot()
%matplotlib inline

df11.CHEMBARAMBAKKAM.resample('M').mean().plot()