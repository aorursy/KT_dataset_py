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
survival_unemp = pd.read_csv("/kaggle/input/survival-unemployment/survival_unemployment.csv")

print(survival_unemp.head())

survival_unemp.describe()

!pip install lifelines
from lifelines import KaplanMeierFitter
# Initiating the KaplanMeierFitter model

kmf = KaplanMeierFitter()
# Spell is referring to time 

T = survival_unemp.spell
# Fitting KaplanMeierFitter model on Time and Events for death 

kmf.fit(T,event_observed=survival_unemp.event)
kmf.plot()