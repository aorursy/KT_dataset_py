# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print('Files available include:')

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





cereal=pd.read_csv('../input/cereal.csv')

print('Shape of cereal: ',cereal.shape)

print(cereal.info())

_=plt.figure()

_=plt.subplot(3,1,1)

_=sns.countplot(cereal['mfr'])

_=plt.subplot(3,1,2)

_=sns.countplot(cereal['shelf'])

_=plt.subplot(3,1,3)

_=sns.countplot(cereal['vitamins'])

aa=plt.subplot(3,1,1)

bb=plt.title('Distribution of categorical data')











plt.clf()

cc=sns.violinplot(cereal['mfr'],cereal['sugars'],scale='count')

plt.clf()

dd=sns.boxplot(cereal['mfr'],cereal['calories'])
plt.clf()

dd=sns.swarmplot('mfr','calories',hue='shelf',data=cereal)


