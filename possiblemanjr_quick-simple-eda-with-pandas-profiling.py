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

import pandas_profiling

family_data = pd.read_csv("../input/santa-workshop-tour-2019/family_data.csv")

family_data.head(5)
family_data.profile_report(style={'full_width':True})
for i in list(family_data.columns):

    print(family_data[i].describe())
for col, values in family_data.iteritems():

    num_uniques = values.nunique()

    print ('{name}: {num_unique}'.format(name=col, num_unique=num_uniques))

    print (values.unique())

    print ('\n')