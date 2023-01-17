# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/multiple-data-nn"))



# Any results you write to the current directory are saved as output.
submission = pd.read_csv("../input/multiple-data-nn/submission__nn__0.9005821511281362.csv")
submission.head()
submission['target'] = submission['target'] * 12 /8
submission.head()
submission.to_csv('output.csv',index=False)