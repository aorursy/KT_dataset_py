# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#dataset = pandas.read_csv("../input/titanic", "r", engine="python");    # will lead to Error: "../input/titanic" is a directory
dataset = pandas.read_csv("../input/titanic/gender_submission.csv", "r");
## Can be used in case entire data has to be checked for, and isn't large enough
# print(dataset)
## Checking initial (5, by default) data
dataset.head()
## Checking ending (5, by default) data
dataset.tail()
## Checks for simple numerical stats (results in tabular form)
dataset.describe()