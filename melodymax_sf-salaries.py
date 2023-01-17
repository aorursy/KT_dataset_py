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

import matplotlib.pyplot as plt


sal = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv', na_values='Not Provided')

sal.head()
sal.info()
sal.describe()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal.loc[sal['TotalPayBenefits'].idxmax()]['EmployeeName']
sal[sal['JobTitle']=='GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY'].mean()
sal.loc[sal['TotalPayBenefits'].idxmin()]
sal.groupby('Year').mean()['BasePay']

sal.groupby('Year').mean()['BasePay'].plot(kind='bar')
sal.groupby('Year').mean()['TotalPayBenefits']
sal.groupby('Year').mean()['TotalPayBenefits'].plot(kind='bar')
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
sal['JobTitle'].apply(lambda str:('chief' in str.lower())).sum()