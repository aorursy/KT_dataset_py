# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%%time
import sys
!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path
!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

import cudf
%%time

train = cudf.read_csv("../input/riiid-test-answer-prediction/train.csv")

print("Train size:", data.shape)
data.head()
questions = cudf.read_csv("../input/riiid-test-answer-prediction/questions.csv")
lectures = cudf.read_csv("../input/riiid-test-answer-prediction/lectures.csv")
example_test = cudf.read_csv("../input/riiid-test-answer-prediction/example_test.csv")
example_sample_submission = cudf.read_csv("../input/riiid-test-answer-prediction/example_sample_submission.csv")
questions.head()
print("questions size:", questions.shape)
lectures.head()
print("lectures size:", lectures.shape)
example_test.head()
print("example_test size:", example_test.shape)
example_sample_submission.head()
print("example_sample_submission size:", example_sample_submission.shape)