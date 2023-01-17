# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import chisquare

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cereal.csv')

data.head(15)
data['fat'].value_counts()
data['protein'].value_counts()
A = data.fat < 4

B = data.protein < 5

C = A & B
modified_data = data[C]
print(chisquare(modified_data.fat,modified_data.protein))
sns.countplot(modified_data.fat)
sns.countplot(modified_data.protein)