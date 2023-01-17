# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import seaborn as sb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cereal_data = pd.read_csv("../input/cereal.csv")
cereal_data.head()
cereal_data.describe()
pd.unique(cereal_data.type)
sb.distplot(cereal_data.carbo).set_title('Carbos Distribution')
sb.distplot(cereal_data.calories).set_title('Calories Distribution')
print('The mean value for calroies is: ', np.average(cereal_data.calories))

print('The mean value for carbs is: ', np.average(cereal_data.carbo))

print('The standard deviation value for calroies is: ', np.std(cereal_data.calories))

print('The standard deviation value for carbs is: ', np.std(cereal_data.carbo))
ttest_ind(cereal_data.calories, cereal_data.carbo, equal_var=False)