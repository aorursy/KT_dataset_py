# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
lines_in_train_set = 74180464
subset_lines = 20000

skip = random.sample(range(1, lines_in_train_set+1), lines_in_train_set - subset_lines)
print(len(skip))
#train_data = pd.read_csv('train.csv', skiprows=skip)
train_data = pd.read_csv('../input/train.csv', skiprows=skip)
head(train_data)