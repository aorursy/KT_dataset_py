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
test_data = pd.read_csv("../input/test.csv")

#產生一個長度跟test data一樣的序列，50%的機率是0，50%的機率是1
rand_labels = (np.random.rand(len(test_data['PassengerId'])) > 0.5).astype(np.int32)

results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : rand_labels
})

results.to_csv("submission1.csv", index=False)
