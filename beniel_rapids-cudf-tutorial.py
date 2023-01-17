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
!nvidia-smi
# Check Python Version

!python --version
# Check CUDA/cuDNN Version

!nvcc -V && which nvcc
import sys

!cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf

from cudf.core import Series

import math
%time a = Series([9, 16, 25, 36, 49], dtype=np.float64)
a
%time a.applymap(lambda x : x**2)
%time a.applymap(lambda x: 1 if x < 18 else 2)
#defining averaging of sqare roots rolling window function



def foo(A):

    sum = 0

    for a in A:

        sum = sum + math.sqrt(a)

    return sum / len(A)
#defining averaging rolling window function



def foo2(A):

    sum = 0

    for a in A:

        sum = sum + a

    return sum / len(A)
%time a.rolling(3, 1, False).apply(foo)
%time a.rolling(3, 1, False).apply(foo2)
%time a.rolling(3, 1, True).apply(foo2)