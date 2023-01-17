# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fpath = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'

data = pd.read_csv(fpath)



fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'

submission = pd.read_csv(fpath)
data.n_people.sum()
data.n_people.hist(bins=7)
import numpy as np

import pandas as pd

#define when to visit

family_day=[] #the day to visit Satan for each family

days_n_ppl=pd.DataFrame({'ppl':np.zeros(100)}) #temporal df



#assign each family to the day which has smallers number of people

for i in range (5000):

    target_index=data.iloc[i,:].iloc[1:11]-1

    target=days_n_ppl.iloc[target_index,:][days_n_ppl.iloc[target_index,:]==days_n_ppl.iloc[target_index,:].min()].dropna() #pickup the nth date which has minimum aggregated n_people

    

    #add family once the day is fixed for the family

    days_n_ppl[days_n_ppl.index==target.index[0]]=days_n_ppl[days_n_ppl.index==target.index[0]]+data.iloc[i,11]

    family_day.append(target.index[0]+1)

   

submission=pd.DataFrame({'family_id':np.arange(0,5000),

                         'assigned_day':family_day})

submission.assigned_day=submission.assigned_day+1

submission.to_csv(f'submission.csv',index=False)