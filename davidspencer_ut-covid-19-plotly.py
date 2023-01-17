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
import plotly.express as px



df = pd.read_csv("/kaggle/input/utah-covid19-2020-03-26/UT-COVID_Cases - Copy of Sheet1.csv")



df.head()

fig = px.scatter(df, x= "date", y= "UT_cases_total")

fig.show()
fig = px.bar(df, x= "date", y= "UT_cases_onset")

fig.show()
fig = px.scatter(df, x= "date", y= "UT_deaths_total")

fig.show()
fig = px.bar(df, x= "date", y= "UT_deaths_daily")

fig.show()