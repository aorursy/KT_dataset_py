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
import matplotlib.pyplot as plt
data = {
'year': [2010, 2011, 2012,
2010, 2011, 2012,
2010, 2011, 2012],
'team': ['FCBarcelona', 'FCBarcelona', 'FCBarcelona',
'RMadrid', 'RMadrid', 'RMadrid',
'ValenciaCF', 'ValenciaCF', 'ValenciaCF'],
'wins': [30, 28, 32, 29, 32, 26, 21, 17, 19],
'draws': [6, 7, 4, 5, 4, 7, 8, 10, 8],
'losses': [2, 3, 2, 4, 2, 5, 9, 11, 11]
}
football = pd.DataFrame(data, columns = ['year', 'team', 'wins', 'draws', 'losses'])
edu = pd.read_csv('/kaggle/input/ict-lesson/files/ch02/educ_figdp_1_Data.csv',
                  na_values=':', usecols=['TIME', 'GEO', 'Value'])
edu

edu.head()
edu.tail()

edu[10:15]
edu.describe()
edu['Value']
edu['GEO']
edu.iloc[90:94][['TIME','GEO']]
edu.max(axis = 0)
