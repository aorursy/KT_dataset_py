# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
bins = [col for col in train.columns if 'bin' in col]

print (bins)
# checking combination of 4 columns

for bin1, bin2, bin3, bin4 in list(itertools.combinations(bins, 4)):

        df = train[[bin1, bin2, bin3, bin4]]

        if len(df.sum(axis=1).unique())<=2:

            print (bin1, bin2, bin3, bin4, df.sum(axis=1).unique())           
# checking combination of 3 columns

for bin1, bin2, bin3 in list(itertools.combinations(bins, 3)):

        df = train[[bin1, bin2, bin3]]

        if len(df.sum(axis=1).unique())<=2:

            print (bin1, bin2, bin3, df.sum(axis=1).unique())           
for bin1, bin2 in list(itertools.combinations(bins, 2)):

        df = train[[bin1, bin2]]

        if len(df.sum(axis=1).unique())<=2:

            print (bin1, bin2, df.sum(axis=1).unique())   