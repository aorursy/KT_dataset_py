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

import numpy as np

from pandas_profiling import ProfileReport
df = pd.read_csv('/kaggle/input/aircrash-data/Airplane_Crashes_and_Fatalities_Since_1908.csv')

df.head(3)
df.describe(include='all')
profile = ProfileReport(df, title = "Flight Detail", html = {'style':{'full_width':True}})

profile
profile.to_notebook_iframe()
profile.to_file(output_file="Flight_report.html")