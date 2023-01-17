# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/corona-virus-report/worldometer_data.csv'

df = pd.read_csv(path)
df.head(3)
#function to get the first digit of the number sequence

def first_digit(x):

    return int(str(x)[0])
#for sanity

df.drop_duplicates('Country/Region', inplace=True)
#Total cases first digit frequency and first digit percentate occurance

totcases = df['TotalCases'].apply(lambda x: first_digit(x))

totcases_vc = totcases.value_counts().sort_index()

totcases_vc_per = (totcases_vc/len(df)* 100)
#Total population first digit frequency and first digit percentate occurance

totpop = df[~df['Population'].isna()]['Population'].apply(lambda x: first_digit(x))

totpop_vc = totpop.value_counts().sort_index()

totpop_vc_per = (totpop_vc/len(df)* 100)
#Benford's Law pattern in COVID-19 TotalCases

y_l = ['Frequency', 'Percentage']



f, a = plt.subplots(1, 2, figsize = (16, 8), 

                    frameon=False)



for i, v in enumerate([totcases_vc, totcases_vc_per]):

    a[i].plot(range(1, 10), v)

    a[i].set_xlabel('First Digit')

    a[i].set_ylabel(y_l[i])

plt.style.use('fivethirtyeight')

plt.show()

    
#Benford's Law pattern in COVID-19 World Population

y_l = ['Frequency', 'Percentage']



f, a = plt.subplots(1, 2, figsize = (16, 8), 

                    frameon=False)



for i, v in enumerate([totpop_vc, totpop_vc_per]):

    a[i].plot(range(1, 10), v)

    a[i].set_xlabel('First Digit')

    a[i].set_ylabel(y_l[i])

plt.style.use('fivethirtyeight')

plt.show()