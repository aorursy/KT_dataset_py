# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('https://www.kaggle.com/anikannal/solar-power-generation-data?select=Plant_1_Generation_Data.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv') 
df1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df.head(10),df1.head(10)


df.info(),df1.info()
from sklearn.model_selection import train_test_split
len(df),len(df1)
train, test = train_test_split(df, train_size=0.9, random_state=10)
train1, test1 = train_test_split(df1, train_size=0.9, random_state=10)

# train, test = train_test_split(df, test_size=0.1)
print(len(train),len(train1))
print(len(test),len(test1))
train.head(),train1.head()
test.head(), test1.head()
df.columns, df1.columns
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df1["DATE_TIME"] = pd.to_datetime(df1["DATE_TIME"])
df = pd.merge(df1,df, on="DATE_TIME", how="inner")
df
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#df[["SOURCE_KEY"]].groupby("SOURCE_KEY").count().index
plt.ylim(150000,250000)
plt.xlim(1400,2100)
plt.title("SOURCE_KEY yield total")
plt.xlabel("time units")
plt.ylabel("total yield")
initial = {}

   
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=10)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, 
                                                    test_size=0.1, 
                                                    random_state=10)
X_train.head(),X_train1.head()
X_test.head(),X_test1.head()
y_train.head(),y_train1.head()
y_test.head(),y_test1.head()