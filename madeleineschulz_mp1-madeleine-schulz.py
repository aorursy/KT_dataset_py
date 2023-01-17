# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#I used data from https://data.census.gov/cedsci/table?q=98122&hidePreview=false&tid=ACSST5Y2018.S1101&vintage=2018&layer=zcta5&cid=DP05_0001E&g=8600000US98122

#I was able to load it into Kaggle, but it is not easy to understand or analyze.
import pandas as pd

data = pd.read_csv("../input/98122-commuting-data/ACSST5Y2018.S0802_data_with_overlays_2020-02-06T124521.csv")
data.head()
data.T
data.drop("(X)")