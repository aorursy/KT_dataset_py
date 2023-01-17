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
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
train.head()
train.Dates.describe()
train.Category.describe()
train.Category.unique()
train.Descript.describe()
train.DayOfWeek.describe()
train.PdDistrict.describe()
train.PdDistrict.value_counts()
train.Resolution.describe()
train.Resolution.value_counts()
train.Address.describe()
train.X.describe()
train.Y.describe()
# some thing wrong with the latitude, need to get rid of wrong data to keep normalization work
