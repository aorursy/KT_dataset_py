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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
plt.rc('figure', figsize=(15, 5))
insurance = pd.read_csv('/kaggle/input/insurance-premium-prediction/insurance.csv')
age = insurance.age
pd.Series({'Count Age': age.count(), 

           'Count Missing Age': age.isna().sum(), 

           'Minimum Age': age.min(), 

           'Maximum Age': age.max()})
age.hist();
sns.distplot(age);
pd.Series({'Mean Age': round(age.mean(), 2), 

           'Median Age': age.median(), 

           'Mode Age': list(age.mode())})
(age.max() + age.min()) / 2
age.hist(bins=10, density=True, cumulative=True, histtype='step');
pd.Series({'Range': age.max() - age.min(), 

           'Variance': age.var(), 

           'Standard Deviation': age.std(), 

           'Mean Absolute Definition': age.mad(), })
age.plot.box(vert=False);
pd.Series({'Skewness': round(age.skew(), 2), 

           'Kurtosis': round(age.kurtosis(), 2)})