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
# source NoteBooks :

# https://www.kaggle.com/kailex/m5-forecaster-v2

# https://www.kaggle.com/kyakovlev/m5-dark-magic
data1 = pd.read_csv('../input/blenddata2/M5  Darkmagic.csv').sort_values(by = 'id').reset_index(drop=True)

data2 = pd.read_csv('../input/blenddata/M5 ForecasteR v2.csv').sort_values(by = 'id').reset_index(drop=True)

submission = data1.copy()



for i in range(1,29):

    data1['F'+str(i)] *= 1.02



for c in submission.columns :

    if c != 'id' :

        submission[c] = 0.33*data1[c] + 0.67*data2[c]

        

submission.to_csv('submission.csv',index=False)