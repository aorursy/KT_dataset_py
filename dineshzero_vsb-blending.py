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
out1=pd.read_csv('../input/vsb-competition-base-neural-network/submission.csv')
out2=pd.read_csv('../input/5-fold-lstm-attention-fully-commented/submission.csv')
out1['5-fold']=out2['target']
out1['target2']=out2.target
out1['target']=pd.Categorical(out1['target'])

out1['target2']=pd.Categorical(out1['target2'])
pd.crosstab(out1.target,out1.target2)
out2['target']=pd.to_numeric(out1['target'])+pd.to_numeric(out1['target2'])
out2.loc[out2['target']==2,'target']=1
out2['target'].value_counts()
out2.to_csv('blending_addition.csv',index=False)
out3=out2.copy()
out3['target']=pd.to_numeric(out1['target'])*pd.to_numeric(out1['target2'])
out3.loc[out3['target']==-1,'target']=0
out3.target.value_counts()
out3.to_csv('blending_multiplication.csv',index=False)