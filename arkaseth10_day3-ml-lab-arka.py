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
#Q1

import pandas as pd

df = pd.DataFrame({'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});

print(df)
#2

import pandas as pd

import numpy as np



exam_data  = {'name': ['Arka', 'Ankan', 'Aritra', 'Abhirup', 'Anik', 'Joy', 'Arpan', 'Abhishek', 'Subhrajyoti', 'Anirban'],

        'score': [14, 10, 11.5, 10, np.nan, 21, 13.5, np.nan, 8, np.nan],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes']}

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']



df = pd.DataFrame(exam_data , index=labels)

print(df)
#Q3

import pandas as pd

import numpy as np



exam_data  = {'name': ['Arka', 'Ankan', 'Aritra', 'Abhirup', 'Anik', 'Joy', 'Arpan', 'Abhishek', 'Subhrajyoti', 'Anirban'],

        'score': [14, 10, 11.5, 10, np.nan, 21, 13.5, np.nan, 8, np.nan],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes']}

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']



df = pd.DataFrame(exam_data , index=labels)

print("First three rows of the data frame:")

print(df.iloc[:4])
#4

import pandas as pd

import numpy as np



exam_data  = {'name': ['Arka', 'Ankan', 'Aritra', 'Abhirup', 'Anik', 'Joy', 'Arpan', 'Abhishek', 'Subhrajyoti', 'Anirban'],

        'score': [14, 10, 11.5, 10, np.nan, 21, 13.5, np.nan, 8, np.nan],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes']}

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

df = pd.DataFrame(exam_data , index=labels)

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6], [1, 3]])
#5

import pandas as pd

import numpy as np

exam_data  = {'name': ['Arka', 'Ankan', 'Aritra', 'Abhirup', 'Anik', 'Joy', 'Arpan', 'Abhishek', 'Subhrajyoti', 'Anirban'],

        'score': [14, 10, 11.5, 10, np.nan, 21, 13.5, np.nan, 8, np.nan],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes']}

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']



df = pd.DataFrame(exam_data , index=labels)

print("Rows where score is missing:")

print(df[df['score'].isnull()])