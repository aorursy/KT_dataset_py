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
#3 variables are created (String,Int,Float)
variable1="String"
variable2=34
variable3=4.0
#Sample list is created
list_set=[1,2,3,4,5,6]
#Sample data dictionary is created
data_dict={'Value1': 100, 'Value2': 200, 'Value3': 300, 'Value4':400}
#Now with single statement, you will be able to see all the variables created globally across the notebook, data type and data/information
%whos