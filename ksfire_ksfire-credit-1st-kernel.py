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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# File system manangement
import os
print(os.listdir("../input"))

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder


## NO  # Suppress warnings 
## NO  import warnings
## NO  warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

ksf_train = pd.read_csv('../input/application_train.csv')
print(ksf_train.shape,"ksf_train.shape")

bureau_mon_loan_bal = pd.read_csv('../input/bureau_balance.csv')
print(bureau_mon_loan_bal.shape,"bureau_mon_loan_bal.shape")

POS_CASH_bal = pd.read_csv('../input/POS_CASH_balance.csv')
print(POS_CASH_bal.shape,"POS_CASH_bal.shape")

prev_appl = pd.read_csv('../input/previous_application.csv')
print(prev_appl.shape, "prev_appl.shape")

instal_pmt = pd.read_csv('../input/installments_payments.csv')
print(instal_pmt.shape, "instal_pmt.shape")

cc_bal = pd.read_csv('../input/credit_card_balance.csv')
print(cc_bal.shape, "cc_bal.shape")

bureau_prevLoans = pd.read_csv('../input/bureau.csv')
print(bureau_prevLoans.shape, "bureau_prevLoans.shape")

ksf_test = pd.read_csv('../input/application_test.csv')
### my ANSWER  " target "  ###
print(ksf_test.shape, "ksf_test.shape")
POS_CASH_bal.head()




bureau_mon_loan_bal.head()

prev_appl.head()

instal_pmt.head()

cc_bal.head()

bureau_prevLoans.head()