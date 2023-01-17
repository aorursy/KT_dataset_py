# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").split()[1].strip()

print(filename)

df = pd.read_csv("../input/" + filename, engine = 'python', sep = '\t' , thousands=",")
df.head()
df.dtypes
df['AnimalBirthMonth'].head()
df['LicenseIssuedDate'].head()
df['LicenseExpiredDate'].head()
df['LicenseExpiredDate'] = pd.to_datetime(df['LicenseExpiredDate'])
df['LicenseIssuedDate'] = pd.to_datetime(df['LicenseIssuedDate'])
df['AnimalBirthMonth'] = pd.to_datetime(df['AnimalBirthMonth'])
df.describe()
df['AnimalName'].value_counts()
df['AnimalName'].value_counts().head(10).keys()
df.groupby('AnimalGender')['AnimalName'].value_counts()