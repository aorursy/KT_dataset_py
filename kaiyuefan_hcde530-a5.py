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

trump = pd.read_csv("../input/trump-popularity/approval_topline.csv")
trump['approve_estimate'].head()

trump.plot(kind='line', x='modeldate',y='approve_estimate')
trump = trump[::-1]

trump.plot(kind='line', x='modeldate',y='approve_estimate')
# x = trump[['modeldate']]

# y = trump[['approve_estimate', 'disapprove_estimate']]

trump[['modeldate', 'approve_estimate', 'disapprove_estimate']].plot(x='modeldate')
trump.plot(kind='line', x='modeldate',y=['approve_estimate', 'disapprove_estimate'])
def pop(approve_estimate): 

    if approve_estimate < 40:

        return "Poor"

    elif approve_estimate > 40 and approve_estimate<=45:

        return "Average"

    elif approve_estimate >45:

        return "Stellar"

trump['approve_estimate'].apply(pop).value_counts().plot(kind='barh')