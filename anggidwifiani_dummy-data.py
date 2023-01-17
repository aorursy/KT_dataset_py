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
# import library

import pandas as pd



# read dataset

data = pd.read_csv('/kaggle/input/scl-dummy/Dummy data.csv')



#make copy for new dataset

data_update = data.copy()
data_update
lambda x: x+1
data_update['number'] = data_update['number'].apply(lambda x: x+1)
data_update.head()
data_update.columns = ['id', 'new_number']
data_update.head()
#make new data

data_update.to_csv(r'/kaggle/working/submission.csv', index = False)