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
# import libraries
# load datasets


# your code here
# write result
a = None

# your code here
b = None

# your code here
# write result
c = None

# your code here
d = None

# your code here
e = None

# your code here
f = None

results = pd.Series(
    {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f
    }, name='results'
)

results.index.name = 'key'
results.to_csv('/kaggle/working/results.csv')
