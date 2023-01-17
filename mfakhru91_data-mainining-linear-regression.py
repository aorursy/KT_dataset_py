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
# Import library

import pandas  as pd #Data manipulation

import numpy as np #Data manipulation

import matplotlib.pyplot as plt # Visualization

import seaborn as sns #Visualization

plt.rcParams['figure.figsize'] = [8,5]

plt.rcParams['font.size'] =14

plt.rcParams['font.weight']= 'bold'

plt.style.use('seaborn-whitegrid')
df = pd.read_csv("../input/insurance/insurance.csv") 

print('\nNumber of rows and columns in the data set: ',df.shape)

print('')

df.head()
sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)

plt.xlabel('Boby Mass Index $ (kg / m ^ 2) $: sebagai variabel Independen')

plt.ylabel('Biaya Asuransi: sebagai variabel Tanggungan')

plt.title('Charge Vs BMI');