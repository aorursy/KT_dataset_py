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

World_Development_Indicators = pd.read_csv("../input/World_Development_Indicators.csv")
type(WDI.iloc[0,:])
import numpy as np

WDI = pd.read_csv("../input/World_Development_Indicators.csv")
type(WDI.iloc[0,:])
WDI.head()
pd.value_counts(WDI["Country Name"],sort=True)
US=WDI[WDI["Country Name"]=="United States"]
China=WDI[WDI["Country Name"]=="China"]

Japan=WDI[WDI["Country Name"]=="Japan"]

S_Korea=WDI[WDI["Country Name"]=="Korea, Rep."]

Germany=WDI[WDI["Country Name"]=="Germany"]

pd.value_counts(WDI["Series Name"])
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
year=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018] # establish year vector

USCO2=US.iloc[[4],4:] # US CO2 emission