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
train = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/train.csv')

test = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/test.csv')
train['no_medboat'] = train['MedBoat'].isnull()

test['no_medboat'] = test['MedBoat'].isnull()
train.corr()
predictions = test['no_medboat'].map({True:0,False:1})
submission = pd.read_csv('/kaggle/input/data-science-nigeria-ai-in-citie/sample_submission.csv')
submission['Survived'] = predictions
submission.to_csv('sub.csv', index=False)