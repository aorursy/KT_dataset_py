# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = {}
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fname = os.path.join(dirname, filename)
        k = f"{dirname.split('/')[len(dirname.split('/'))-1]}_{filename.split('.')[0]}"
        data[k] = pd.read_csv(fname)
        print(fname)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#
data.keys()
data['the-movies-dataset_links'].head()