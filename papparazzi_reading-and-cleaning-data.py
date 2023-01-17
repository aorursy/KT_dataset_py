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
# read data

full_table = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])



# combine china and mainland china

full_table['Country/Region'].replace({'China':'Mainland China'},inplace=True)

countries = full_table['Country/Region'].unique().tolist()



# filling missing values

full_table[['Province/State']] = full_table[['Province/State']].fillna('--')



print("\nTotal countries affected by CoVID-19 thus far: ",len(countries))

print("\nLatest daily case reports included in the notebook: ", str(max(full_table['Date']))[:10])
display(full_table)