# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import scipy.stats as sts

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

cerealData = pd.read_csv('../input/cereal.csv')

cerealData.head()

# Any results you write to the current directory are saved as output.
import seaborn as sns

sns.distplot(cerealData['calories'], kde = False).set_title('Calories Distibution')
hotCereal = cerealData.loc[cerealData['type'] =='H','sugars']

coldCereal= cerealData.loc[cerealData['type'] =='C','sugars']

ttest_ind(hotCereal, coldCereal, equal_var = False)

# p value : 0.01874

sns.distplot(hotCereal, kde = False,bins=20,color='R').set_title('Sugars Distibution in hot cereal')

sns.distplot(coldCereal, kde = False,bins=20,color='G').set_title('Sugars Distibution in cold Cereal')
sns.countplot(x='type',data=cerealData).set_title('Hot and Cold Cereals')
sns.countplot(x='type',hue='mfr',data=cerealData).set_title('Hot and Cold Cereals based on Manufacturer')
sts.chisquare(cerealData.carbo)