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
import pandas as pd #imports Python Data Analysis Library



df = pd.read_csv('../input/scl-dummy/Dummy data.csv')

df.head()
num = [[i, i + 2] for i in range(101)] #formula for the contents of the table (i + 2)

df = pd.DataFrame(num, columns = ['id', 'new_number']) #creates the table

df.describe()
df.to_csv('submission.csv', header = True, index = False) #saves the csv file