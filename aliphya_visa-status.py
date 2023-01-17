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
!pip install jovian --upgrade -q
import jovian
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

df = pd.read_csv("/kaggle/input/h-1b-visa/h1b_kaggle.csv")
df.head()

df.rename(columns={ 'Unnamed: 0' : 'Index' }, inplace = 'True')
df.info()
df.describe()
df["CASE_STATUS"].drop_duplicates()
df.dropna()

jovian.commit(project = "Visa_status", files = ['h1b_kaggle.csv'])

df_certified = df[(df["CASE_STATUS"]=='CERTIFIED')]
df_certified
jovian.commit(project = "Visa_status")
df_certified.groupby('YEAR').count().plot()
df_certified.groupby('YEAR').count()
jovian.commit(project = "Visa_status")
df_certified[df_certified["FULL_TIME_POSITION"] == "Y"].count()
df_certified['FULL_TIME_POSITION'].hist()
jovian.commit(project = "Visa_status")
df.info()
df.groupby('CASE_STATUS').count()
df.groupby('CASE_STATUS').count().plot(kind = 'bar')
df_denied = df[df['CASE_STATUS'] == 'DENIED']
df_denied
#df_denied.rename(columns={ 'Unnamed: 0' : 'Index' }, inplace = 'True')
df_denied.groupby('YEAR').count().plot()
df_denied_wage = df_denied['PREVAILING_WAGE']
df_denied_count = df_denied['CASE_STATUS'].count()
df_denied_avg = ((df_denied_wage.sum())/df_denied_count)


print("Avg wage of visa denied candidates:",  df_denied_avg)

df_certified_wage = df_certified['PREVAILING_WAGE']
df_certified_count = df_certified['CASE_STATUS'].count()
df_certified_avg = ((df_certified_wage.sum())/df_certified_count)


print("Avg wage of visa certified candidates:",  df_certified_avg)

jovian.commit(project = "Visa_status")
df_withdrawn = df[(df["CASE_STATUS"]=='WITHDRAWN')]
df_withdrawn

df_withdrawn.groupby('SOC_NAME').count().sort_index(ascending=False).head(10).plot(kind = "bar")
df_withdrawn.groupby('SOC_NAME').count().sort_index(ascending=False).head(10)
jovian.commit(project = "Visa_status")
jovian.commit(project = "Visa_status")