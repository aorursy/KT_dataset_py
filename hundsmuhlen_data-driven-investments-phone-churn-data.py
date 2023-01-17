# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
sns.set(style="black")
sns.set(style="blackgrid", color_codes=True)

import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = data.dropna()

print(data.shape)
data.head()
data = data.replace({'Churn': {'No': 0, 'Yes': 1}}).replace({'PaperlessBilling': {'No': 0, 'Yes': 1}}).replace({'PhoneService': {'No': 0, 'Yes': 1}}).replace({'Dependents': {'No': 0, 'Yes': 1}}).replace({'Partner': {'No': 0, 'Yes': 1}}).replace({'gender': {'Male': 0, 'Female': 1}})
data.head()
data['gender'].unique()
data.rename(columns={"gender": "Female"})
data['Churn'].value_counts()