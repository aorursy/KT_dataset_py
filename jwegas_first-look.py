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
category = pd.read_csv('../input/Category.csv')
print (category.info(), '\n', category.shape, '\n', category.head())
Location = pd.read_csv('../input/Location.csv')
print (Location.info(), '\n', Location.shape, '\n', Location.head())
ItemPairs_train = pd.read_csv('../input/ItemPairs_train.csv')
print(ItemPairs_train.info(), '\n', ItemPairs_train.shape, '\n', ItemPairs_train.head())
ItemPairs_test = pd.read_csv('../input/ItemPairs_test.csv')
print(ItemPairs_test.info(), '\n', ItemPairs_test.shape, '\n', ItemPairs_test.head())
ItemInfo_train = pd.read_csv('../input/ItemInfo_train.csv', encoding='utf-8')
print(ItemInfo_train.info(), '\n', ItemInfo_train.shape, '\n', ItemInfo_train.head())
ItemInfo_test = pd.read_csv('../input/ItemInfo_test.csv')
print(ItemInfo_test.info(), '\n', ItemInfo_test.shape, '\n', ItemInfo_test.head())
ItemInfo_test = pd.read_csv('../input/ItemInfo_test.csv')
print(ItemInfo_test.info(), '\n', ItemInfo_test.shape, '\n', ItemInfo_test.head())
Random_submission = pd.read_csv('../input/Random_submission.csv')
print(Random_submission.info(), '\n', Random_submission.shape, '\n', Random_submission.head())