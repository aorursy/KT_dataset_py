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
df = pd.read_excel(r'../input/ty-entc-grades/ty_entc.xlsx')
df.head()
print("Mean SGPA is", df['SGPA'].mean(axis=0))
dfMPPR = df['MPPR']
df.MPPR.mode()
dfMPPR.value_counts()
#function to generate registration number
def generator(val):
    return '2017bec0'+str(val)

#generating registration number of students in A2 Batch
for i in range(31,66):
    if i%2 == 1:
        a = generator(i)
        b = df.loc[df['Registration no'] == a]
        print(b)
df.plot(x='RANK',y="SGPA")
df.hist()
df.loc[df['Name'] == 'Siddhesh Mahajan'].head() #fuck chetan
