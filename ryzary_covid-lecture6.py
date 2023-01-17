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
import pandas as pd

import numpy as np
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
#let's set ObservationDate as the index and drop the unneeded columns

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()
data_NZ = data[data['Country/Region']=='New Zealand']

data_NZ.tail()
#Ranking by 'Confirmed' case

latest = data[data.index=='06/02/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



#New Zealand's Ranking

print('Ranking of New Zealand: ', latest[latest['Country/Region']=='New Zealand'].index.values[0]+1)