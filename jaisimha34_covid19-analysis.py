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
import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt 
case = pd.read_csv('../input/coronavirusdataset/case.csv')

patient = pd.read_csv('../input/coronavirusdataset/patient.csv')

route = pd.read_csv('../input/coronavirusdataset/route.csv')

time = pd.read_csv('../input/coronavirusdataset/time.csv')

trend = pd.read_csv('../input/coronavirusdataset/trend.csv')
case.head()
case = case.drop('case_id', axis=1)
patient.head()
patient.drop('patient_id', axis = 1 )
plt.figure(figsize=(30,20))



sns.countplot(patient['region'])
case.head()
sns.scatterplot(x='longitude',y='latitude', data= case, hue= 'confirmed')
sns.countplot(case['province'])