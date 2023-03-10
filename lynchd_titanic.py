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
# Read in all csv files

gender_submission = pd.read_csv('../input/gender_submission.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
# Sanity Check the Files

print('Test Model \n', test.head())

print('Train Model \n', train.head())

print(train.info())

print(train.describe())

print(gender_submission.head())