import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

bureau_application = pd.read_csv('../input/application_train.csv')

# CREDIT BUREAU
# bureau: previous credits by financial institutions
bureau = pd.read_csv('../input/bureau.csv')

# bureau_balance: history of payment for each credit in bureau
bureau_balance = pd.read_csv('../input/bureau_balance.csv')

bureau_join = pd.merge(
    bureau,
    bureau_balance,
    on = 'SK_ID_BUREAU',
    how = 'inner'
)

example = bureau_join[bureau_join['SK_ID_CURR'].isin([380361,399518,215382,401081])]
example_sort = example.sort_values(['SK_ID_CURR','SK_ID_BUREAU','MONTHS_BALANCE'], ascending = True)
example_sort.to_csv('Curious_loans_plus.csv', index = False)
