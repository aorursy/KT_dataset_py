# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
data = data.dropna()
price = data['Volume_(Currency)']/data['Volume_(BTC)']
rent = np.log(price/price.shift(1)).dropna()
pnl = 0

pos = 0

for r in rent:

    pnl = pnl + pos*np.exp(r)

    pos = -r

    pnl = pnl + r

     

print(pnl)