import pandas as pd

import scipy as sp

from scipy.stats import ttest_ind as TTest

import seaborn as sns

import matplotlib.pyplot as plt



path = "../input/HR Exit Velocity.csv"

df = pd.read_csv(path)

df_2015 = df[df['Season'] == 2015]

df_2017 = df[df['Season'] == 2017]



TTest(df_2017['HR Exit Velocity'],df_2015['HR Exit Velocity'])
df15hr = df_2015['HR Exit Velocity']

df17hr = df_2017['HR Exit Velocity']



f, axes = plt.subplots(2, figsize=(15, 10), sharex=True, sharey=True)



sns.distplot(df15hr, bins=200, kde=False, color = "b", ax=axes[0]).set_title("2015")

sns.distplot(df17hr, bins=75, kde=False, color = "r", ax=axes[1]).set_title("2017")
df15hr.describe()
df17hr.describe()