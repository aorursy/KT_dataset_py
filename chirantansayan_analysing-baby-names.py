# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
names = pd.read_csv(r'/kaggle/input/us-national-baby-names-18802017/all_national.csv')
names.head()
names.info()
names.groupby('Sex').Count.sum()
total_birth = names.pivot_table(index='Year',columns='Sex',aggfunc=sum)

total_birth.tail()
plt.figure(figsize=(20,9))

total_birth.plot()
boys = names[names.Sex == 'M']

girls = names[names.Sex == 'F']

total_b = names.pivot_table('Count', index = 'Year', columns = 'Name', aggfunc=sum)

total_b
subset = total_b[['Jhon','Harry','Mary','Marylin']]

subset.plot()
names.plot(style={'M':'k-','F':'k--'})