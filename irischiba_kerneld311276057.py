# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
c=pd.read_csv('../input/class3/class3.csv',header=None)

#submission.to_csv('submission.csv',index=False)

c.columns=['AdoptionSpeed']

c['AdoptionSpeed']=pd.to_numeric(c['AdoptionSpeed']).astype(np.int32)

c.head()
sample_sub=pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')

sample_sub.head()



sub=pd.concat([sample_sub['PetID'],c['AdoptionSpeed']],axis=1)



len(sub)
sub.to_csv('submission.csv',index=False)

sub.head()