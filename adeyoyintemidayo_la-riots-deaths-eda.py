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
data = pd.read_csv("../input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv")
## import the seaborn package 

import seaborn as sns

## import the matplotlib package 

from matplotlib import pyplot as plt 
## check the data

data.head()
## check info

data.info()
## check for missing values 

data.isnull().sum()
## handling missing values 

data["Solved*"].fillna(("Unsolved"), inplace = True)
## handling missing values 

data["lat"].fillna((data["lat"].mean()), inplace = True )

data["lon"].fillna((data["lon"].mean()), inplace = True )
## drop the URL 

data.drop("URL", axis = 1 , inplace = True)
data.isnull().sum()
data.shape
data.columns
## lets check the dataframe

data.head()
sns.countplot(data["Gender"])
data["Gender"].value_counts()
## plotting the race by gender 

sns.countplot(data["Race"],hue=data["Gender"])
## status count

data["status"].value_counts().plot(kind = "bar")
data["status"].value_counts()
## race count

data["Race"].value_counts().plot(kind = "bar")
data["Race"].value_counts()