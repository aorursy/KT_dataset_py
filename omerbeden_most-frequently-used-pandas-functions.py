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
os.chdir("/kaggle/input/novel-corona-virus-2019-dataset")
import pandas as pd
covid_df = pd.read_csv("covid_19_data.csv")
covid_df
covid_df.head(10)
last_update_date = covid_df['Last Update']
last_update_date = list(map(lambda str:str.split()[0],last_update_date))
covid_df[covid_df['ObservationDate']==last_update_date]
print(covid_df['ObservationDate'].tail(10))
print(covid_df['Last Update'].tail(10))
covid_df.describe()
covid_df.index
covid_df.dtypes
covid_df.axes
covid_df.columns
covid_df.keys()
covid_df.size
covid_df.shape
covid_df['Country/Region'].unique()
covid_df[covid_df['Deaths']==covid_df['Deaths'].max()]
covid_df[covid_df['Deaths']<50]['Country/Region'].iloc[:30]
covid_df.isna().any()
covid_df.isnull().any()
covid_df.notnull().all()
covid_df.fillna(value=0)
covid_df.dropna()
covid_df.drop_duplicates()
covid_df.drop(['SNo'],axis=1)
covid_df.select_dtypes(include=['object'])
covid_df.select_dtypes(exclude=['object'])
covid_df.select_dtypes(include=['int'])
covid_df['Deaths'].value_counts()
covid_df.sort_values(by=['Deaths'],ascending=False)
covid_df.groupby(['Country/Region']).mean().iloc[:,1:]
covid_df.corr(method='pearson')