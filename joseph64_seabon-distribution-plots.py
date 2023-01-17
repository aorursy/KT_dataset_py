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
import seaborn as sns #data visalization

%matplotlib inline 
# we load our tips dataset from seaborn

tips = sns.load_dataset('tips')

# we can check the first 5 values of our dataset

tips.head()
sns.distplot(tips['total_bill'])

# we can create our dist plot with the code above 

# the output should be a histogram and a kde plot
# we can get rid of the kde plot by adding the code kde = False

sns.distplot(tips['total_bill'], kde = False)
