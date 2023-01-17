import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/application_train.csv')

installments_payments = pd.read_csv('../input/installments_payments.csv')
installments_payments_staging = installments_payments \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in installments_payments_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in installments_payments_staging.columns.levels[1][:-1]:
            columns.append('installments_payments_%s_%s' % (var, stat))

installments_payments_staging.columns = columns
train = pd.merge(
    train,
    installments_payments_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)
installments_payments_corr_staging = train[columns]
installments_payments_corr_staging['TARGET'] = train['TARGET']

installments_payments_corr = installments_payments_corr_staging.corr()
print(installments_payments_corr['TARGET'].sort_values(ascending = False))

#credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean       0.107692
#credit_card_balance_CNT_DRAWINGS_CURRENT_max            0.101389
#credit_card_balance_AMT_BALANCE_mean                    0.087177
#credit_card_balance_AMT_TOTAL_RECEIVABLE_mean           0.086490