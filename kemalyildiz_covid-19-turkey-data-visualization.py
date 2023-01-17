# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cvData_df = pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
cvData_df.head()
cvData_df.info()
cvData_df['Active'] = cvData_df["Confirmed"]-cvData_df["Recovered"]-cvData_df["Deaths"]
cvData_df.drop(columns="Province/State")
cvData_df['Last_Update'] = pd.to_datetime(cvData_df['Last_Update'])
cvData_df.info()
time_df = cvData_df.groupby('Last_Update')['Confirmed','Deaths','Recovered', 'Active'].sum()
print(time_df.head())

print(type(time_df))
plt.figure(figsize=(16, 8))

sns.lineplot(data=time_df)
plt.xlabel("Time")
plt.ylabel("Activity")
plt.show() 
cvData_df.corr()
corrMatrix = time_df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
