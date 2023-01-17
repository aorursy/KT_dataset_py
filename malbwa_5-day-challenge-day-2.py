# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(check_output(["ls", "../input/individual_stocks_5yr"]).decode("utf8"))
all_stocks_1yr = pd.read_csv('../input/all_stocks_1yr.csv')

all_stocks_1yr.head()
aapl_5yr = pd.read_csv('../input/individual_stocks_5yr/AAPL_data.csv')

aapl_5yr.head()
aapl_5yr.describe()
sns.distplot(aapl_5yr['Close']).set_title('AAPL Close')