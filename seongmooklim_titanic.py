# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#Print the `head` of the train and test dataframes

test_one = test.copy()

test_one['Survived'] = 0



test_one['Survived'][test_one['Sex'] == 'femail'] = 1

test_one['Survived'][test_one['Sex'] == 'male'] = 0



print(train.head())

print(test.head())

print(test_one['Survived'])