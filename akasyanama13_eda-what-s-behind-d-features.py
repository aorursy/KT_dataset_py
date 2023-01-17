import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
by = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

grouped = train.groupby(by, as_index=False)['TransactionID'].count()

grouped[grouped['TransactionID']==7].head(5)
# This combination of cardx features gives 7 rows.

card1 = 18383

card2 = 128

card3 = 150

card4 = 'visa'

card5 = 226

card6 = 'credit'



train_slice = train[(train['card1']==card1)&

                   (train['card2']==card2)&

                   (train['card3']==card3)&

                   (train['card4']==card4)&

                   (train['card5']==card5)&

                   (train['card6']==card6)]
features = ['TransactionID','TransactionDT','ProductCD', 'P_emaildomain', 'R_emaildomain'

            , 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']

train_slice = train_slice.sort_values(['TransactionID'])[features]

train_slice
train_slice['DaysFromStart'] = np.round(train_slice['TransactionDT']/(60*60*24),0)
train_slice
train_slice['DaysFromPreviousTransaction'] = train_slice['DaysFromStart'].diff()
features = ['TransactionID', 'TransactionDT', 'D1', 'D2', 'D3', 'DaysFromStart', 'DaysFromPreviousTransaction']

train_slice[features].iloc[3:]