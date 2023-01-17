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
import pandas as pd
import numpy as np
df_test = pd.read_csv('../input/test.csv')
PassengerId=np.array(df_test['PassengerId']).astype(int)
all_zeros=np.zeros(418,dtype=int)
my_solution=pd.DataFrame(all_zeros, PassengerId, columns = ['Survived'])
my_solution.to_csv('all0.csv',index_label=['PassengerId'])

