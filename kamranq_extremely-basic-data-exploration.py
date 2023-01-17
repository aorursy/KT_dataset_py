# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
cats = [x for x in data.columns if data[x].dtype.name == 'object']

print('\nCategorical:\n',cats)

nums = [x for x in data.columns if data[x].dtype.name != 'object']

print('\nNumeric:\n',nums)
missingCat = {}

for level in cats:

    if data[level].isnull().sum()>0:

        missingCat[level] = data[level].isnull().sum()

print('Categorical:\n', missingCat)

print('\nNumber of Categorical columns with missing values: ',len(missingCat))



missingNum = {}

for level in nums:

    if data[level].isnull().sum()>0:

        missingNum[level] = data[level].isnull().sum()

print('\n\nNumeric:\n', missingNum)

print('\nNumber of Numeric columns with missing values: ',len(missingNum))
cat_level = {}

for level in cats:

    cat_level[level] = data[level].nunique()

print(cat_level)