# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from scipy.stats import ttest_ind

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data  = pd.read_csv('../input/7210_1.csv')

data.describe()

# columns colour and price#

#

purpleShoePrices =data['prices.amountMin'][data['colors'] == 'Purple']

otherShoePrices =data['prices.amountMin'][data['colors'] != 'Purple']



#data['prices.amountMin]

#purpleShoePrice = data.prices[ind]

t = ttest_ind(purpleShoePrices,otherShoePrices, equal_var=False)

print(t)

# Any results you write to the current directory are saved as output.
import seaborn as sns

sns.distplot(purpleShoePrices,kde=False).set_title('purple shoes')





sns.distplot(otherShoePrices,kde=False).set_title('other shoes')


