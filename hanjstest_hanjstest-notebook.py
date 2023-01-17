# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import random

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")





res_data = pd.DataFrame(columns=['installation_id', 'accuracy_group'])







res_data['installation_id'] = data['installation_id'].drop_duplicates()

res_data['accuracy_group'] = random.randint(0, 3)





res_data.to_csv("/kaggle/working/submission.csv", index=False)



res_data