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
test = pd.read_csv("../input/test.csv")

test_one = test.copy()
test_one['Survived'] = 0;

test_one['Survived'][test_one['Sex'] == 'female'] = 1
test_one['Survived'][test_one['Sex'] == 'male'] = 0

print(test_one['Survived'])
df_submission = pd.DataFrame({
    "PassengerId": test_one["PassengerId"],
    "Survived": test_one['Survived']
})
df_submission.to_csv('./submission_all_rf_10th.csv', index=False)
