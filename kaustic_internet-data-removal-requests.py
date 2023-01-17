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
data = pd.DataFrame(

    {

        "Term": pd.Series([], dtype=object),

        "Country": pd.Series([], dtype=object),

        "count": pd.Series([], dtype=np.int32),

    }

)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        df_temp = pd.read_csv(os.path.join(dirname, filename))

        df_temp["Term"] = np.full(df_temp.shape[0], filename[-11:-4])

        data = data.append(df_temp, ignore_index=True)

        
data.isnull().value_counts()
data.info()
del data["Unnamed: 2"]
# The one null value was the total



data = data.dropna()
data.groupby("Country")["count"].sum().sort_values(ascending=False).head(20)
data.loc[data["count"].idxmax()]
import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use("seaborn-darkgrid")
mean_count_summary = data[data["count"] > 200].groupby(["Country", "Term"]).agg(

    COUNTRY=("Country", "max"), MEAN_REQUESTS=("count", "mean")

)



plt.figure(figsize=(12, 10))



sns.boxplot(x="MEAN_REQUESTS", y="COUNTRY", data=mean_count_summary)



plt.xlabel("Mean of Requests per Term")

plt.ylabel("Country")



plt.legend()

plt.show()
plt.figure(figsize=(12, 8))



sns.barplot(

    "Term",

    "count",

    data=data[data["Country"] == "India"]

)



plt.show()
plt.figure(figsize=(16, 8))



total_reqs_by_country = data.groupby("Country").agg(total=("count", "sum"), country=("Country", "max"))

total_reqs_by_country = total_reqs_by_country[total_reqs_by_country["total"] >= 1000]



sns.barplot("country", "total", data=total_reqs_by_country)



plt.xlabel("Country")

plt.ylabel("Total Requests")



plt.show()