# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load data :
file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
data = pd.read_csv(file_path)
data.head()
#Answer 2
data.sample(n=10)
#Answer 3
data.shape
data.describe
data.dtypes
data.isnull().any()
#Answer 4
data.isnull().sum()
data.isnull().head()
data['children'].unique()
data['babies'].unique()
data['country'].unique()
data['agent'].unique()
data['company'].unique()
#Answer 5
data["children"].fillna(0,inplace=True)
data["children"].isnull().any()
#Answer 6
data["country"].fillna(method="ffill",inplace=True)
data["country"].isnull().any()
da=data.columns[data.isnull().any()]
da
#Answer 7
data["agent"].fillna(data["agent"].mean(),inplace=True)
data["agent"].isnull().any()
#Answer 8
numberofhotel=data["hotel"].unique()
len(numberofhotel)
numberofcustomers=data["customer_type"].unique()
len(numberofcustomers)
plt.boxplot(data["children"])
plt.boxplot(data["agent"])