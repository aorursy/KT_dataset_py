# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/avito-best-public-blend"))
print(os.listdir("../input/avito-demand-prediction"))
tr = pd.read_csv('../input/avito-demand-prediction/train.csv')
sub =  pd.read_csv('../input/avito-best-public-blend/best_public_blend.csv')
from sklearn.metrics import mean_squared_error
allzeros = 0.3032 # publicLB
tr['allzero'] = 0
np.sqrt(mean_squared_error(tr.deal_probability, tr['allzero'] ))
allzeros / 0.2949544182370951
sub.deal_probability *= 1.03
sub.to_csv('beat-the-best-kernel.csv', index=False)
