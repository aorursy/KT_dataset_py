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
#Importing the dataset
train=pd.read_csv('/kaggle/input/bank-predicting-credit-application-response/Credit_Application_Response.csv')

#Exploring the train data
train.head()
#Exploring the train data null values and data types
train.info()
#Data Exploration in Single Step is acheived with the help of Pandas_Profiling package
import pandas_profiling as pdf
from pandas_profiling import ProfileReport
#Installing the package of Pandas Profiling
!pip install pandas_profiling
#Create a summary report with Pandas Profiling

pdf.ProfileReport(train)

#From the below report that is produced, you can toggle between different variables and other statistics produced.