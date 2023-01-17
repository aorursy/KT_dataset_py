# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
import seaborn as sns

sns.set(rc={'figure.figsize':(15,12)})
sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.ProductCD), annot = True, fmt = "d")
sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.card4), annot = True, fmt = "d")
sns.heatmap(pd.crosstab(train_transaction_df.isFraud, train_transaction_df.card6), annot = True, fmt = "d")
sns.heatmap(pd.crosstab(train_transaction_df.P_emaildomain, train_transaction_df.isFraud), annot = True, fmt = "d")
sns.heatmap(pd.crosstab(train_transaction_df.R_emaildomain, train_transaction_df.isFraud), annot = True, fmt = "d")
sns.heatmap(pd.crosstab(train_transaction_df.M1, train_transaction_df.isFraud), annot = True, fmt = "d")
drp_row = train_transaction_df.dropna()
drp_row.shape
drp_col = train_transaction_df.dropna(axis = 'columns')
drp_col.shape
drp_col.head()
drp_col.nunique()
crstb = pd.crosstab(drp_col.card1, drp_col.isFraud)
crstb
crstb_two = crstb.unstack().reset_index().rename(columns={0:"cnt"})
crstb_two
crstb_two.sort_values(by=['isFraud', 'cnt'], ascending=False)
import seaborn as sns
sns.regplot(x = crstb_two.card1, y = crstb_two.isFraud, logistic=True)