import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/application_train.csv')

credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
credit_card_balance_staging = credit_card_balance \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in credit_card_balance_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in credit_card_balance_staging.columns.levels[1][:-1]:
            columns.append('credit_card_balance_%s_%s' % (var, stat))

credit_card_balance_staging.columns = columns
train = pd.merge(
    train,
    credit_card_balance_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)
credit_card_balance_corr_staging = train[columns]
credit_card_balance_corr_staging['TARGET'] = train['TARGET']

credit_card_balance_corr = credit_card_balance_corr_staging.corr()
print(credit_card_balance_corr['TARGET'].sort_values(ascending = False))

#credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean       0.107692
#credit_card_balance_CNT_DRAWINGS_CURRENT_max            0.101389
#credit_card_balance_AMT_BALANCE_mean                    0.087177
#credit_card_balance_AMT_TOTAL_RECEIVABLE_mean           0.086490