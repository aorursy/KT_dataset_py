# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sub = pd.read_csv("../input/bert-pred1/bert_prediction.csv")

sub = sub.iloc[:,1:]



sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
id_in_sub = set(sub.qa_id)

id_in_sample_submission = set(sample_submission.qa_id)

diff = id_in_sample_submission - id_in_sub



sample_submission = pd.concat([

    sub,

    sample_submission[sample_submission.qa_id.isin(diff)]

]).reset_index(drop=True)
sub
sample_submission.head()
sub.to_csv("submission.csv", index=False)
# Adding random values

#eps = np.random.random_integers(low=0, high=0.001, size=temp.iloc[:,1:].shape)

#temp.iloc[:,1:] = temp.iloc[:,1:]+eps

"""

min_max = MinMaxScaler(feature_range=(0.01, 0.99))

temp.iloc[:,1:] = min_max.fit_transform(temp.iloc[:,1:])



temp.sort_values(by='qa_id',axis=0,inplace=True)

temp = temp.reset_index(drop=True)

temp.to_csv("submission.csv",index=False,float_format= '%.20f')"""
"""temp"""
"""temp.iloc[:,1:].min()"""
"""temp.iloc[:,1:].max()"""