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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
path = "../input/coronavirusdataset/Time.csv"



corona_data = pd.read_csv(path,index_col="date", parse_dates=True)
corona_data
plt.figure(figsize=(12,10))

plt.title("Monthly test corona ")

sns.lineplot(data=corona_data)

# Add label for horizontal axis

plt.xlabel("Date of test") # Your code here

plt.ylabel("The results of the tests") # Your code here
path = "../input/coronavirusdataset/TimeAge.csv"



corona_data2 = pd.read_csv(path,index_col="date", parse_dates=True)
corona_data2
plt.figure(figsize=(12,10))

plt.title("Age of Corona sufferers")

sns.barplot(y=corona_data2.confirmed, x=corona_data2.age)

#sns.heatmap(data=corona_data, annot=True)

plt.xlabel("Date") # Your code here
plt.figure(figsize=(12,10))

plt.title("Age of Corona sufferers")

sns.barplot(y=corona_data2.deceased, x=corona_data2.age)

#sns.heatmap(data=corona_data, annot=True)

plt.xlabel("Date") # Your code here
path = "../input/coronavirusdataset/TimeGender.csv"



corona_data3 = pd.read_csv(path,index_col="date", parse_dates=True)
corona_data3
plt.figure(figsize=(12,10))

plt.title("Monthly Visitors to Avila Adobe")

sns.barplot(x=corona_data3.sex, y=corona_data3.confirmed)

plt.xlabel("Corona infected cases") # Your code here

plt.ylabel("The number of infected cases") # Your code here