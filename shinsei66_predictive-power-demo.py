!pip install ppscore
import numpy as np 

import pandas as pd 

import ppscore as pps

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PATH ='/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(f'{PATH}')

df.shape
list(df.columns)
df.dtypes
%%time

pps.score(df, 'InternetService', 'Churn')
%%time

pps.score(df, 'PaymentMethod', 'Churn')
%%time

df_matrix = pps.matrix(df)
plt.figure(figsize=(18,18))

sns.heatmap(df_matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

plt.show()