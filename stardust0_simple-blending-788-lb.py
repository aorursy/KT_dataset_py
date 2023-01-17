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
data1 = pd.read_csv('../input/lightgbm-with-simple-features-0-785-lb/submission_kernel00.csv')
data2 = pd.read_csv('../input/tidy-xgb-all-tables-0-782/tidy_xgb_0.77821.csv')
data1['TARGET'] = (data1['TARGET']+data2['TARGET'])/2
data1.to_csv('blend1_.788lb.csv',index = False)
