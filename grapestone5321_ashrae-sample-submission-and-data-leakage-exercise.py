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
df = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
df.head(10)
df.to_csv('submission.csv', index = False)
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex7 import *

print("Setup Complete")
q_1.solution()
q_2.solution()
q_3.solution()
q_4.solution()
q_5.hint()
q_5.solution()