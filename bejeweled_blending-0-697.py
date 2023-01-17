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
from scipy.stats import rankdata
sub_files = [

    "../input/new-blend/ens_sub_v2.csv",

    "../input/new-blend/sub.csv",

    "../input/new-blend/super_blend.csv",

    "../input/newsub/submission.csv",

]
df1 = pd.read_csv(sub_files[0])

df2 = pd.read_csv(sub_files[1])

df3 = pd.read_csv(sub_files[2])

df4 = pd.read_csv(sub_files[3])
ens_sub = pd.read_csv("../input/microsoft-malware-prediction/sample_submission.csv")
ens_pred = np.array(df1['HasDetections']).std()*df1['HasDetections'] + np.array(df2['HasDetections']).std()*df2['HasDetections'] +np.array(df3['HasDetections']).std()*df3['HasDetections'] + np.array(df4['HasDetections']).std()*df4['HasDetections']
ens_pred = rankdata(ens_pred, "dense")
ens_pred.std()
ens_sub['HasDetections'] = ens_pred
ens_sub.HasDetections *= 1+ens_sub.HasDetections.std()
ens_sub.HasDetections -= ens_sub.HasDetections.min()/2
ens_sub.head()
ens_sub.to_csv("ens_sub_v16.csv", index=False)