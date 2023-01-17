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
ccb = pd.read_csv('../input/credit_card_balance.csv')
ccb.head()
ccb.columns = ['이전id','현재id','잔액갱신여부','잔액','한도','인출금액ATM','인출금액','인출금액기타','인출금액POS','월간최소할부금','전월카드사용액','전월총카드사용액','미수금원금','미수금','총미수금','ATM인출건수','인출건수','기타인출건수','POS인출건수','이전할부지불횟수','계약상태','연체일수','연체일수_소액대출']
ccb.head()
from collections import Counter as co
ccb2 = ccb.fillna(0)
ccb2[ccb2['인출금액'] != ccb2['인출금액ATM'] + ccb2['인출금액POS'] + ccb2['인출금액기타']]
ccb2 = ccb.sort_values(['현재id','잔액갱신여부'])
ccb3 = ccb2.fillna(0)
np.unique(ccb3['계약상태'])
ccb3[ccb3['계약상태']=='Refused']
ccb3[ccb3['현재id']==102367].sort_values('잔액갱신여부')
ccb3[ccb3['인출금액'] != ccb3['인출금액ATM']+ccb3['인출금액POS']+ccb3['인출금액기타']]
