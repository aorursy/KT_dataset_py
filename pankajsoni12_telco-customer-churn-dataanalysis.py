# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head(2)
df = df.drop(["customerID"],axis=1)
df.head(1)
df.Churn.value_counts().plot(kind="bar")
plt.show()
df.gender.value_counts().plot(kind="pie")
plt.show()
df.SeniorCitizen.value_counts().plot(kind="bar")
plt.show()
plt.figure(figsize=(16,4))
df[["Churn","tenure"]][df.Churn=="Yes"].tenure.value_counts().sort_values(ascending=False).plot(kind="bar")
plt.figure(figsize=(16,4))
ch_tn = df[["Churn","tenure"]][df.Churn=="Yes"]
plt.scatter(range(ch_tn.tenure.size),ch_tn.tenure,alpha = .1,color="g",s=70)
plt.show()
# Below graph clearly tells that lesser tenure (upto 15'yrs) customer are more declined towerds departing the service provider 
# Further analysis we can target on those data where churning population is more (like tenure from 1yr to 15'yrs)
plt.figure(figsize=(16,4))
df[df.Churn == "Yes"].tenure.value_counts().plot(kind="bar")
plt.show()
# Below graph clearly tells that lesser tenure customer are more declined towerds departing the service provider 
# Further analysis we can target on those data where churning population is more (like tenure from 1yr to 15'yrs)
df.describe()
df.info()