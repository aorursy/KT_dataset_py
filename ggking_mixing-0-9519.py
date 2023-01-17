# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



sub_1 = pd.read_csv('/kaggle/input/cvopenc/0.9468.csv')



sub_2 = pd.read_csv('/kaggle/input/experimental/0.9468.csv')



sub_3 = pd.read_csv('/kaggle/input/catboos/0.9407.csv')



sub_4 = pd.read_csv('/kaggle/input/r-score/0.9469.csv')



sub_1['isFraud'] += sub_2['isFraud']

sub_1['isFraud'] += sub_3['isFraud']

sub_1['isFraud'] += sub_4['isFraud']



sub_1.to_csv('submission.csv', index=False)

sub_1