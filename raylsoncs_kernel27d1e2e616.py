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
for var in range(0,16):

    df = pd.read_csv('/kaggle/input/kernel52cd1e9b1f/file_' + str(var) +'.csv.gz', header=None)

    

    compression_opts = dict(method='zip', archive_name='CSV_' + str(var) + '.csv')

    

    df.to_csv('file_' + str(var) + '.zip', index=False, header=False, compression=compression_opts)