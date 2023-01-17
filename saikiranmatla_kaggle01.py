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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/covid_19_data.csv")
data.head()
data.info()
data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])
data['Last Update'] = pd.to_datetime(data['Last Update'])
data.head()
data.info()
data_cp = data.copy()
data_cp["month"] = data_cp["ObservationDate"].map(lambda x : x.month)
data_cp["date"] = data_cp["Last Update"].dt.date
data_cp["time"] = data_cp["Last Update"].dt.time
data_cp
month_median = data_cp["month"].median()
print(month_median)
updated = data_cp.copy()
updated
updated1 = updated.set_index(['date','time'])
updated1
updated2 = updated1.drop('Last Update', axis=1)
updated2
updated2 = updated2.sort_index(ascending =True)
updated3 = updated2.loc["2020-06-13"]
updated3


len(updated3["Country/Region"].unique())
updated4 = updated3[["Country/Region","Confirmed","Deaths","Recovered"]]
updated4
top_10 = updated4.sort_values("Confirmed", ascending=False)
top_10.head(10)
top_10["Recovered_percentage"] = top_10["Recovered"]/top_10["Confirmed"] * 100
top_10["Deaths_percentage"] = top_10["Deaths"]/top_10["Confirmed"] * 100
top_10
top_10_sorted = top_10.sort_values("Recovered_percentage",ascending=False)
top_10_sorted
top_10_sorted = top_10.sort_values("Deaths_percentage",ascending=True)
top_10_sorted.head(10)
Germany = data[(data["Country/Region"] == 'Germany')]
Germany
Germany_data = Germany[["ObservationDate","Confirmed","Deaths","Recovered"]]
Germany_data
sns.jointplot(Germany["Confirmed"],Germany["Recovered"])
updated3.describe()
updated5 = updated3.copy()
#updated5
usa = updated5[(updated5["Country/Region"] == "US")]
usa
usa_gp = usa.groupby(["Province/State","Confirmed"]).size()
#usa_gp
usa_gp_df = pd.DataFrame(usa_gp)
usa_gp_df.sort_values("Confirmed", ascending=False)