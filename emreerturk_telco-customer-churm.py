# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head(8)
data.tail(8)
data.info()
data.describe()
data.columns
data["PaymentMethod"]
data.drop(["SeniorCitizen"],axis=1,inplace=True)
data.head()
data.gender=[1 if each=="Male" else 0 for each in data.gender]
data.gender.value_counts()
data.info()
data.PaymentMethod.value_counts()