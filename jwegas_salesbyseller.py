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
train = pd.read_csv('../input/train.csv', nrows=0)
train.head()
train_chunk = pd.read_csv('../input/train.csv', chunksize=100000)
for chunk in train_chunk:
    train = pd.concat([train, chunk[chunk['Agencia_ID'] == 1111]])
train['Demanda_uni_equil']
import matplotlib.pyplot as plt
plt.plot(train['Demanda_uni_equil'][:100], '-')
help(plt.plot)