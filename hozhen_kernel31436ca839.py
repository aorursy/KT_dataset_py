# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from kaggle.competitions import nflrush



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df.shape
train_df.info()
train_df.head()
sns.distplot(train_df.Yards)
sns.kdeplot(train_df.YardLine)
sns.kdeplot(train_df.Distance)
sns.kdeplot(train_df.groupby('GameId').max().Yards)
sns.kdeplot(train_df.groupby('GameId').min().Yards)
sns.kdeplot(train_df.groupby('GameId').median().Yards)
sns.kdeplot(train_df.groupby('GameId').mean().Yards)
a1 = plt.figure()

a1 = sns.FacetGrid(train_df, col = "FieldPosition", row = "PlayDirection")

a1 = a1.map(sns.distplot, 'Yards')
train_df.groupby('FieldPosition').PlayDirection.value_counts()
a1 = plt.figure()

a1 = sns.FacetGrid(train_df, col = "OffenseFormation", col_wrap = 4)

a1 = a1.map(sns.distplot, 'Yards')
mini_df = train_df[['PlayId','YardLine','Yards','FieldPosition','PlayDirection']].drop_duplicates('PlayId').reset_index()
mini_df.info()
mini_df.head()
endog = [i for i in range(-99,100)] 
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional as KDE
gene_dens = KDE(endog = mini_df.Yards, exog = mini_df.YardLine, dep_type='c', indep_type='c', bw='normal_reference')
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    mini_test = test_df[['PlayId','YardLine','FieldPosition','PlayDirection']].drop_duplicates('PlayId')

    Y = mini_test.YardLine.iloc[0]

    FP = test_df.FieldPosition.iloc[0]

    PD = test_df.PlayDirection.iloc[0]

    train_mini_df = mini_df[(mini_df.FieldPosition == FP) & (mini_df.PlayDirection == PD)].reset_index()

    dens = KDE(endog = train_mini_df.Yards, exog = train_mini_df.YardLine, dep_type='c', indep_type='c', bw=gene_dens.bw)

    pred_value = dens.cdf(endog, [Y]*199)

    pred_value[198] = 1

    pred_value[pred_value>1]=1

    pred_value[pred_value<0]=0

    sample_prediction_df.iloc[0] = pred_value

    if sample_prediction_df.iloc[0].isnull().any()==1 :

        pred_value = gene_dens.cdf(endog,[Y]*199)

        pred_value[198] = 1

        pred_value[pred_value>1]=1

        pred_value[pred_value<0]=0

        sample_prediction_df.iloc[0] = pred_value

    env.predict(sample_prediction_df)



env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])