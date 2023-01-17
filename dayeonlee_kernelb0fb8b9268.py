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
train=pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

specs=pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")

sample_submission=specs=pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

train_labels=pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

test=pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
train.head()
specs.head()
train_labels.head()
test.head()