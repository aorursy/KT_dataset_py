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
import pandas as pd

H1 = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

H1

H2 = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

H2

H3 = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

H3

H4 = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

H4

H5 = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

H5

H1.to_csv('submission.csv',index=False)

H2.to_csv('submission.csv',index=False)

H3.to_csv('submission.csv',index=False)

H4.to_csv('submission.csv',index=False)

H5.to_csv('submission.csv',index=False)

#H1.to_csv('submission.csv',index=False)
