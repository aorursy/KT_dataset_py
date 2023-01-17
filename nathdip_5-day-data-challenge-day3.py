# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data_cereal = pd.read_csv('../input/cereal.csv')

data_cereal.describe()

list(data_cereal)

# Any results you write to the current directory are saved as output.
cereal_mfr_name = data_cereal['mfr']

cereal_mfr_name[0:10]
mfr_k = data_cereal.loc[data_cereal['mfr']=='K']

mfr_g = data_cereal.loc[data_cereal['mfr']=='G']
calories_k = mfr_k['calories']

print('The standard deviation of the calorific value of cereal made by K: ', np.std(calories_k))

calories_g = mfr_g['calories']

print('The standard deviation of the calorific value of cereal made by G: ', np.std(calories_g))

import seaborn as sns

sns.distplot(calories_k, kde = False, color='b').set_title('Histogram for Cereals manufactured by K')



sns.distplot(calories_g, kde = False, color='g').set_title('Histogram for Cereals manufactured by G')

from scipy.stats import ttest_ind

ttest_value = ttest_ind(calories_k, calories_g, equal_var = False)

print('The p-value is: ', ttest_value.pvalue)