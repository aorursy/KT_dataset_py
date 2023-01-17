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
test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

# Save test predictions to file

for class_id in range(1,8):

    preds_test = (np.ones(len(test))*class_id).astype(int)



    output = pd.DataFrame({'Id': test.index,

                           'Cover_Type': preds_test})

    output.to_csv('submission-'+str(class_id)+'.csv' , index=False)

    output.head()
count = { 1: 0.37062,

 2: 0.49657,

 3: 0.05947,

 4: 0.00106,

 5: 0.01287, 

 6: 0.02698, 

 7: 0.03238} 

weight = [count[x]/(sum(count.values())) for x in range(1,7+1)]

class_weight_lgbm = {i: v for i, v in enumerate(weight)}