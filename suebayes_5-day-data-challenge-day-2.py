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



df = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')



df.describe()



df['High']

sns.distplot(df['High'], kde = False) .set_title("High Value of BitCoin");





# Any results you write to the current directory are saved as output.