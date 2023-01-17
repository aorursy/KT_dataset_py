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

hc = pd.read_csv("../input/hate_crime_2017.csv", parse_dates = True, infer_datetime_format = True)

hc['INCIDENT_DATE'] = pd.to_datetime(hc['INCIDENT_DATE'])
hc.columns
#figure out number of hate crimes per year

StateHC = hc.groupby(['STATE_NAME', hc['INCIDENT_DATE'].dt.year])['TOTAL_OFFENDER_COUNT'].sum()



arr = []



for state in set(hc['STATE_NAME'].values):

    tmp = []

    for year in range(1992,2018):

        if (state,year) in StateHC:

            tmp.append(StateHC[(state,year)])

    arr.append(tmp)

    print(state, tmp)
set(hc['STATE_NAME'].values)