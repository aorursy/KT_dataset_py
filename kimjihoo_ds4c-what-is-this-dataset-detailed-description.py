# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/coronavirusdataset/'



case = p_info = pd.read_csv(path+'Case.csv')

p_info = pd.read_csv(path+'PatientInfo.csv')

#p_route = pd.read_csv(path+'PatientRoute.csv')

time = pd.read_csv(path+'Time.csv')

t_age = pd.read_csv(path+'TimeAge.csv')

t_gender = pd.read_csv(path+'TimeGender.csv')

t_provin = pd.read_csv(path+'TimeProvince.csv')

region = pd.read_csv(path+'Region.csv')

weather = pd.read_csv(path+'Weather.csv')

search = pd.read_csv(path+'SearchTrend.csv')

floating = pd.read_csv(path+'SeoulFloating.csv')

policy = pd.read_csv(path+'Policy.csv')
case.head()
p_info.head()
#p_route.head()
time.head()
t_age.head()
t_gender.head()
t_provin.head()
region.head()
weather.head()
search.head()
floating.head()
policy.head()