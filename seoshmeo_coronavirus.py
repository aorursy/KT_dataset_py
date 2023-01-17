# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from plotly.offline import iplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
patient = pd.read_csv("/kaggle/input/coronavirusdataset/patient.csv")

route = pd.read_csv("/kaggle/input/coronavirusdataset/route.csv")

time = pd.read_csv("/kaggle/input/coronavirusdataset/time.csv")
patient.country.value_counts()

route.visit.value_counts()
route.city.value_counts()
patinet_sex = patient.sex.value_counts()

patient.sex.value_counts()



patient.infection_reason.value_counts()
year = patient.birth_year

for i in year:

    age = 2019 - year

age_count = age.value_counts()

print(age_count)
plt.xlabel("days") 

plt.ylabel("number_of_conf.") 

plt.grid() 

plt.plot(time.acc_confirmed)

plt.plot(time.new_confirmed )