# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/cereal.csv')

df.describe()

# Any results you write to the current directory are saved as output.
from scipy.stats import ttest_ind



ttest_ind(df['sugars'], df['sodium'], equal_var=False)
import matplotlib.pyplot as plt



plt.title('Sugars histogram')

plt.hist(df['sugars'])



plt.show()



plt.title('Sodium histogram')

plt.hist(df['sodium'])



plt.show()