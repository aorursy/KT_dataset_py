import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
df = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

df1 = df[df.Response == 1]

df0 = df[df.Response == 0]



df_test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
f,ax=plt.subplots(1,1,figsize=(30,10))

title = plt.title('Box plot by Premium', fontsize=20)

title.set_position([0.5, 1.05])



sns.boxplot(x='Response', y='Annual_Premium',data=df,ax=ax)

ax.set_xlabel('Response')

ax.set_ylabel('Annual_Premium')

g = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=15)

plt.show()
f,ax=plt.subplots(1,1,figsize=(30,10))

title = plt.title('Distribution till 3rd quartile', fontsize=20)

title.set_position([0.5, 1.05])



sns.distplot(a=df[df.Annual_Premium < 80000]['Annual_Premium'], kde=False,ax=ax)



plt.show()
df2630 = df[df.Annual_Premium == 2630]

df2630_test = df_test[df_test.Annual_Premium == 2630]

df_non_2630 = df[df.Annual_Premium != 2630]

df_non_2630_test = df_test[df_test.Annual_Premium != 2630]
f,ax=plt.subplots(1,1,figsize=(30,10))

title = plt.title('Box plot by Premium without Premium Amount 2630', fontsize=20)

title.set_position([0.5, 1.05])



sns.boxplot(x='Response', y='Annual_Premium',data=df_non_2630,ax=ax)

ax.set_xlabel('Response')

ax.set_ylabel('Annual_Premium')

g = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=15)

plt.show()
np.quantile(df_non_2630['Annual_Premium'], 0.95)
f,ax=plt.subplots(1,1,figsize=(30,10))

title = plt.title('Distribution without Premium Amount 2630 till 95th quantile', fontsize=20)

title.set_position([0.5, 1.05])



sns.distplot(a=df_non_2630[df_non_2630.Annual_Premium < 57184]['Annual_Premium'], kde=False,ax=ax)



plt.show()