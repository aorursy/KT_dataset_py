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

wage_gender = pd.read_csv("../input/city-of-seattle-wages-comparison-by-gender/City_of_Seattle_Wages___Comparison_by_Gender_-_All_Job_Classifications.csv")
wage_gender[['Female Avg Hrly Rate','Male Avg Hrly Rate','Jobtitle']]
wage_gender[['Female Avg Hrly Rate','Male Avg Hrly Rate','Jobtitle']].plot(x='Jobtitle', kind='bar', figsize=(50,5))
wage_gender['wage rate difference'] = wage_gender['Male Avg Hrly Rate'] - wage_gender['Female Avg Hrly Rate'] 

wage_gender[['wage rate difference','Jobtitle']].plot(x='Jobtitle', kind='bar', figsize=(50,5))
wage_gender.sort_values(by=['wage rate difference'])[['wage rate difference','Jobtitle']].plot(x='Jobtitle', kind='bar', figsize=(50,5))
wage_gender.sort_values(by=['wage rate difference'])[['wage rate difference','Jobtitle']].dropna().plot(x='Jobtitle', kind='bar', figsize=(50,5))

wage_gender.sort_values(by=['wage rate difference'])[['wage rate difference','Jobtitle']].head()
wage_gender.sort_values(by=['wage rate difference'])[['wage rate difference','Jobtitle']].dropna().tail()
import pandas as pd

wage_age = pd.read_csv("../input/city-of-seattle-wages-average-hourly-wage-by-age/City_of_Seattle_Wages__Comparison_by_Gender_-_Average_Hourly_Wage_by_Age.csv")
wage_age[['AGE RANGE','Average of FEMALE HOURLY RATE','Average of MALE HOURLY RATE']].plot(x='AGE RANGE', kind='bar')
wage_age['wage rate difference'] = wage_age['Average of MALE HOURLY RATE'] - wage_age['Average of FEMALE HOURLY RATE']

wage_age[['AGE RANGE','wage rate difference']].plot(x='AGE RANGE', kind='bar')