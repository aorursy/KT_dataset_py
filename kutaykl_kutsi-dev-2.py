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

covid_19 = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")
covid_19.head()
covid_19["Deaths"]


list=[]

for x in covid_19["Deaths"]:

    list.append(x)

list2=np.nan_to_num(list)

maxvalue=max(list2)

print(maxvalue)

theindex=list.index(maxvalue)

print(theindex)



list2=[]

for y in covid_19["Country/Region"]:

    list2.append(y)

print(list2[theindex])