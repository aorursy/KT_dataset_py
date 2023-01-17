# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

zf_train = zipfile.ZipFile('../input/nyc-taxi-trip-duration/train.zip')
data_train = pd.read_csv(zf_train.open('train.csv'), nrows=100)

zf_test = zipfile.ZipFile('../input/nyc-taxi-trip-duration/test.zip')
data_test = pd.read_csv(zf_test.open('test.csv'), nrows=100)

zf_sample_submission = zipfile.ZipFile('../input/nyc-taxi-trip-duration/sample_submission.zip')
data_sample_submission = pd.read_csv(zf_sample_submission.open('sample_submission.csv'), nrows=100)

print(data_train)
print(data_test)
print(data_sample_submission)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.