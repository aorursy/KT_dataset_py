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
import torch

def gradient(u, i):

    u.requires_grad_(True)

    expNumerator = torch.exp(u)

    p = expNumerator[i] / expNumerator.sum()

    intermediateResult = p*p.log()

    Result = (-1)*intermediateResult.sum()

    Result.backward()

    print(u.grad)
gradient(torch.tensor([1.,2.,3.]), 1)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



covertype = pd.read_csv("../input/dataset_184_covertype.csv")



covertype_y = covertype["class"]

covertype_x = covertype.drop("class",axis=1)



label_encoder = LabelEncoder().fit(covertype_y)

cover_target = label_encoder.transform(covertype_y)



df_train, df_test, y_train, y_test = train_test_split(covertype_x, cover_target, test_size=0.15, stratify=cover_target)