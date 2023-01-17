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
train = pd.read_csv('../input/drugsComTrain_raw.csv')
train.head()
train.shape
rev = np.array(train['review'])
rev[:10]
rev[-10:]
drugs = train['drugName'].value_counts()
drugs
# number of drugs with only 1 review
one_rev = len(np.where(drugs==1)[0])
print('Drugs with one review: ', one_rev)
print('Drugs with one review: ', round((one_rev/drugs.shape[0])*100), '%')
cond = train['condition'].value_counts()
cond
# number of conditions with only 1 review
one_rev = len(np.where(cond==1)[0])
print('Conditions with one review: ', one_rev)
print('Conditions with one review: ', round((one_rev/drugs.shape[0])*100), '%')

