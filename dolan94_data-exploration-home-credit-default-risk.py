# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import os
os.listdir('../input')
train_data = pd.read_csv('../input/application_train.csv')
test_data = pd.read_csv('../input/application_test.csv')
#prev_appl = pd.read_csv('../input/previous_application.csv')
#bureau = pd.read_csv('../input/bureau.csv')
cc_balance = pd.read_csv('../input/credit_card_balance.csv')
#POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
#bureau_balance = pd.read_csv('../input/bureau_balance.csv')
#installments_payments= pd.read_csv('../input/installments_payments.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
print("train Data", train_data.shape)
print("test_data", test_data.shape)
print("prev_appl", prev_appl.shape)
print("bureau", bureau.shape)
print("cc_balance", cc_balance.shape)
print("POS_CASH_balance", POS_CASH_balance.shape)
print("bureau_balance", bureau_balance.shape)
print("installments_payments", installments_payments.shape)
print("sample_submission", sample_submission.shape)

bureau[bureau['SK_ID_CURR'] == 215354]
bureau.groupby(['SK_ID_CURR']).count()
print("Unique SK_ID_CURR in bureau", bureau.SK_ID_CURR.unique())
print("since there is only one SK_ID_CURR this data is essentially useless")
bureau.shape

#print("SK_ID_CURR unique values in train Data", train_data.SK_ID_CURR.unique())
#print(train_data.groupby(['SK_ID_CURR']).count())
print("credit card balance shape", cc_balance.shape)

# bureau is meaningless since it has only one SK_ID_CURR
cc_balance.head(5)

import numpy as np

unique_months_balance = np.sort(np.array(cc_balance.MONTHS_BALANCE.unique()))
unique_months_balance


alldfs = [var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]

print(alldfs)
_4
data.head()
data.describe()

data.shape
#test.shape
data.columns.values
train_data
import matplotlib.pyplot as plt
%matplotlib inline
#train_data.hist(figsize = (900, 900))
