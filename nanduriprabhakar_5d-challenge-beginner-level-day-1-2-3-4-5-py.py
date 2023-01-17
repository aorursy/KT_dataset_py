# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy.stats import ttest_ind as ttest

from scipy.stats import chisquare



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
raw_file=pd.read_csv('../input/cereal.csv')
raw_file
raw_file.describe()
numeric=raw_file['calories']
sns.distplot(numeric, kde=True).set_title('Histogram of the Calories data present in the dataset')
test1=raw_file['type'] == 'C'

test2=raw_file['type'] == 'H'
data1=raw_file[test1].calories

data2=raw_file[test2].calories
ttest(data1,data2,0,False)
sns.distplot(data1, kde=True)
sns.distplot(data2,kde=False)
sns.countplot(data1).set_title('bar graph for the calories in C type Cereal')
data3=raw_file[['protein','calories']]
chisquare(data3)