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
data = pd.read_csv('../input/Q4.csv')
data.head()
data['Amount Spent (USD)'] = data['Amount Spent (USD)'].str.replace('$', '')
data['Net Revenue (USD)'] = data['Net Revenue (USD)'].str.replace('$', '')
data
data['Amount Spent (USD)'] = data['Amount Spent (USD)'].astype('float')

data['Net Revenue (USD)'] = data['Net Revenue (USD)'].str.replace(',', '')
data['Net Revenue (USD)'] = data['Net Revenue (USD)'].astype('float')
location = data.groupby('Wonoloers Market')
# location['Amount Spent (USD)'].astype('float')
location['Activated (Done First Job)'].sum()
amnt_spent = location['Amount Spent (USD)'].sum()
revenue = location['Net Revenue (USD)'].sum()
revenue/amnt_spent
location.head()
channel = data.groupby('Channel')
signed_channel
signed_channel = channel['Signed Up'].sum()
rev_channel = channel['Net Revenue (USD)'].sum()
spent_channel = channel['Amount Spent (USD)'].sum()
spent_channel/signed_channel
rev_channel/signed_channel