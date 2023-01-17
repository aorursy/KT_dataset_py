# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cereal=pd.read_csv('../input/cereal.csv')

print('Shape of cereal is: ',cereal.shape)

mfr_c=cereal.mfr.value_counts()

type_c=cereal.type.value_counts()

print(mfr_c)

print(type_c)

mfr_type=pd.crosstab(cereal['mfr'],cereal['type'])

print(mfr_type)

from scipy.stats import chisquare, chi2_contingency



print(chisquare(mfr_c))

print(chisquare(type_c))

print(chi2_contingency(mfr_type))

import seaborn as sns

import matplotlib.pyplot as plt

_=plt.subplot(1,2,1)

_=plt.pie(mfr_c,labels=mfr_c.index,autopct='%1.1f%%',radius=2)

_=plt.axis('equal')

_=plt.title('Cereal Manfacturers (Total = {})'.format(mfr_c.sum()))

_=plt.subplot(1,2,2)

_=plt.pie(type_c,labels=type_c.index,autopct='%1.1f%%')

_=plt.axis('equal')

_=plt.title('Hot versus Cold cereals (Total = {})'.format(type_c.sum()))