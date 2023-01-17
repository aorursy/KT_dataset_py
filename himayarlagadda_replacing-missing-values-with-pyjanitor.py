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
!pip install pyjanitor
import pandas as pd

import janitor as jn



#Sample dataframe

sample_df = pd.DataFrame({ "ProductNumber" : [1, None, 3], "SalesNumber": [20.0, 30.0, None]})

print(sample_df)
"""

The coalesce function in pyjanitor takes a dataframe and a list of columns to consider. This is similar to functionality found in excel and SQL databases. 



It returns the first nonnull value for each row



"""



jn.coalesce(sample_df, columns=["ProductNumber", "SalesNumber"], new_column_name = "Row_Non_Null_Value")





"""

We can replace the missing values with a particular value by using pyjanitor fill_empty function

"""



jn.fill_empty(sample_df, columns=["ProductNumber", "SalesNumber"], value = 10)