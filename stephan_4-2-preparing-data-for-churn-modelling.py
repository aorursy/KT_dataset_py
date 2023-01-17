# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/applied-ml-microcourse-telco-churn'



customer = pd.read_csv('{}/customer.csv'.format(path))

contract = pd.read_csv('{}/contract.csv'.format(path))

customer.head()
contract.head()
customer.info()
contract[contract['customerID'].duplicated()].shape
print(customer[customer['customerID'].duplicated()])

print('Customer file shape is: {}'.format(customer.shape))

print('Contract file shape is: {}'.format(contract.shape))
data = customer.merge(contract, on='customerID')

data.head()
data['churn'] = np.where(data['EndDate'].isna(), 0, 1)

print('Merged data has shape: {}'.format(data.shape))

data.head()