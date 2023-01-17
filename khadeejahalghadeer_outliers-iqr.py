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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

Data=pd.read_csv("/kaggle/input/weight-height/weight-height.csv")

Data.head()

#Scatter plot to show the outliers

plt.figure(figsize=(20,8))

plt.scatter(Data.Weight,Data.Height)

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()
Data.describe()
#Height Plot Distrbution

plt.figure(figsize=(20,8))



plt.subplot(1,3,1)

sns.distplot(Data.Height)

plt.ylabel('Distrbution')



plt.subplot(1,3,2)

sns.boxplot(y=Data.Height)



plt.subplot(1,3,3)

plt.hist(Data.Height, bins=20, rwidth=0.8)

plt.xlabel('Height')

plt.ylabel('Count')



plt.show()
Q1=Data.Height.quantile(0.25)

Q3=Data.Height.quantile(0.75)

Q1,Q3
IQR=Q3-Q1

IQR
lower_limit=Q1-1.5*IQR

Upper_limit=Q3+1.5*IQR

lower_limit,Upper_limit
#Height outliers

Data_outlier=Data[(Data.Height<lower_limit)|(Data.Height>Upper_limit)]

Data_outlier
Data_no_outlier=Data[(Data.Height>lower_limit)&(Data.Height<Upper_limit)]

Data_no_outlier
#Weight Plot Distrbution

plt.figure(figsize=(20,8))



plt.subplot(1,3,1)

sns.distplot(Data.Weight)

plt.ylabel('Distrbution')



plt.subplot(1,3,2)

sns.boxplot(y=Data.Weight)



plt.subplot(1,3,3)

plt.hist(Data.Weight, bins=20, rwidth=0.8)

plt.xlabel('Weight')

plt.ylabel('Count')

plt.show()
Q1=Data.Weight.quantile(0.25)

Q3=Data.Weight.quantile(0.75)

Q1,Q3
IQR=Q3-Q1

IQR
Weight_lower_limit=Q1-1.5*IQR

Weight_Upper_limit=Q3+1.5*IQR

Weight_lower_limit,Weight_Upper_limit
# Weight outliers

Data_outlier=Data[(Data.Weight<Weight_lower_limit)|(Data.Weight>Weight_Upper_limit)]

Data_outlier
Data_no_outlier=Data[(Data.Weight>Weight_lower_limit)&(Data.Weight<Weight_Upper_limit)]

Data_no_outlier