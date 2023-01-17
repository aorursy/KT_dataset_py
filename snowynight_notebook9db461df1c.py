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
import os

os.getcwd()

train_data = pd.read_csv("../input/train.csv")
train_data.describe()
train_data.head(5)
pd.isnull(train_data).any()
pd.isnull(train_data).sum()
pearson = train_data.corr(method='pearson')

pearson
corr_with_target = pearson.iloc[-1][:-1]

corr_with_target_dict = corr_with_target.to_dict()

corr_with_target_dict
sorted_dict = sorted(corr_with_target_dict.items(), key = lambda x: -abs(x[1]))

sorted_dict