import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/application_train.csv')

previous_application = pd.read_csv('../input/previous_application.csv')
previous_application = pd.get_dummies(previous_application)
previous_application_staging = previous_application \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in previous_application_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in previous_application_staging.columns.levels[1][:-1]:
            columns.append('previous_application_%s_%s' % (var, stat))

previous_application_staging.columns = columns

print(train.shape)
train = pd.merge(
    train,
    previous_application_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

print(train.shape)
previous_application_corr_staging = train[columns]
previous_application_corr_staging['TARGET'] = train['TARGET']

previous_application_corr = previous_application_corr_staging.corr()
print(previous_application_corr['TARGET'].sort_values(ascending = False))

#previous_application_CODE_REJECT_REASON_XAP_mean                           -0.073930
#previous_application_NAME_CONTRACT_STATUS_Approved_mean                    -0.063521
#previous_application_NAME_CONTRACT_STATUS_Refused_mean                      0.077671
#previous_application_NAME_CONTRACT_STATUS_Refused_sum                       0.064469
#previous_application_CODE_REJECT_REASON_SCOFR_max                           0.063657
#previous_application_NAME_PRODUCT_TYPE_walk-in_sum                          0.062628