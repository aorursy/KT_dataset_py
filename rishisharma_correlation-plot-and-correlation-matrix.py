

#### Importing Libraries ####



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



dataset = pd.read_csv("../input/churn_data.csv")





#### EDA ####





dataset.head(5) # Viewing the Data

dataset.columns

dataset.describe() # Distribution of Numerical Variables



# Cleaning Data

dataset[dataset.credit_score < 300]

dataset = dataset[dataset.credit_score >= 300]



# Removing NaN

dataset.isna().any()

dataset.isna().sum()

dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])





dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',

                    'web_user', 'app_web_user', 'ios_user',

                    'android_user', 'registered_phones', 'payment_type',

                    'waiting_4_loan', 'cancelled_loan',

                    'received_loan', 'rejected_loan', 'zodiac_sign',

                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]



dataset2=dataset2.drop(columns = ['housing', 'payment_type',

                         'registered_phones', 'zodiac_sign'])
dataset2.head()
## Correlation with Response Variable

color=tuple(["g", "b","r","y","k"])

dataset2.corrwith(dataset.churn).plot.bar(figsize=(20,10),

              title = 'Correlation with Response variable',

              fontsize = 15, rot = 45,

              grid = True,color=color)
plt.figure(figsize=(20,10))

sns.heatmap(dataset2.corr(),annot=True)