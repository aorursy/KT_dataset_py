import pandas as pd 

import numpy as np

train = pd.read_csv('../input/train.csv')

members = pd.read_csv('../input/members.csv')



print(train.shape)

print(members.shape)
import functools

dfs = [train,members]

trainBIG = functools.reduce(lambda left,right: pd.merge(left,right,on='msno', how='inner'), dfs)

trainBIG.head()
churn = pd.crosstab(trainBIG['gender'],trainBIG['is_churn'])

print(churn)
churn_rate = churn.div(churn.sum(1).astype(float),

                             axis=0) # normalize the value



churn_rate.plot(kind='barh', stacked=True)
churn2 = pd.crosstab(trainBIG['registered_via'],trainBIG['is_churn'])

print(churn2)
churn_rate2 = churn2.div(churn2.sum(1).astype(float),

                             axis=0) # normalize the value



churn_rate2.plot(kind='barh', stacked=True)
churn3 = pd.crosstab(trainBIG['city'],trainBIG['is_churn'])

print(churn3)
churn_rate3 = churn3.div(churn3.sum(1).astype(float),

                             axis=0) # normalize the value



churn_rate3.plot(kind='barh', stacked=True)
#eliminating extreme outliers

trainBIG = trainBIG[trainBIG['bd'] >= 1]

trainBIG = trainBIG[trainBIG['bd'] <= 100]



import seaborn as sns

sns.boxplot(x=trainBIG["is_churn"],y=trainBIG["bd"],data=trainBIG)