# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Copyright Nolan BNake



# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at



#     http://www.apache.org/licenses/LICENSE-2.0



# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import warnings



warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Add Internal Blend (0.9522)

sub_1 = pd.read_csv('/kaggle/input/09522/submission2.csv')



# Blend of two kernels with old features (0.9524)

sub_2 = pd.read_csv('/kaggle/input/gmean-of-low-correlation-lb-0-952x/stack_gmean.csv')
# Scores give others))

score_1 = 0.95220

score_2 = 0.95240



# Sum

sum_scores = score_1 + score_2 



# Weights with which we should sum all things up.

weight_1 = score_1/sum_scores

weight_2 = score_2/sum_scores





sub_1['isFraud'] = weight_1*sub_1['isFraud'] + weight_2*sub_2['isFraud'] 



# Attention: scores are similar => effect of approach is small, but positive (should be)!

sub_1.to_csv('submission-.9525.csv', index=False)