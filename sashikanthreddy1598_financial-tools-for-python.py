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
import numpy as np

 

cashflows       = [-1000000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000]; # t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

discountRate    = 0.15; # Fifteen percent per annum

npv             = np.npv(discountRate, cashflows);  

 

print("Net present value of the investment:%3.2f"%npv);
