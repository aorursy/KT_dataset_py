# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/legislators.csv')

data.head()
list(data.columns)
data['dateofbirth'] = pd.to_datetime(data['birthdate'])
list(data.columns)
def calculate_age(born): 

    today = datetime.today() 

    return today.year - born.year
data['age']= data['dateofbirth'].apply(calculate_age)
dems = data['party']=='D'

below45= data['age'] < 45



youngDems = data[dems & below45]
youngDems.head()
repub = data['party']=='R'

twitter= data['twitter_id'].isnull() == False

youtube= data['youtube_url'].isnull() == False

republican_social = data[repub & twitter & youtube]
republican_social.head()
youngDems.iloc[:,0:29].to_excel('Young_Democrats.xlsx')

republican_social.iloc[:,0:29].to_excel('republican_socialMedia.xlsx')