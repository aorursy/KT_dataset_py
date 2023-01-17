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
gpu_df = pd.DataFrame.from_csv('../input/All_GPUs.csv')

cpu_df = pd.DataFrame.from_csv('../input/Intel_CPUs.csv')
gpu_df
gpu_df.info()
cpu_df 
import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')
freq = cpu_df['Processor_Base_Frequency']
freq.dropna()
freqs = []



for val in freq.values.tolist():

    freqs.append(val)

freqs = [float(value.split()[0]) if type(value)==str else 0 for value in freqs ]
freq_arr = np.array(freqs)
cpu_df['BaseFreq'] = freq_arr
qtrs = cpu_df['Launch_Date'].tolist()
qtrs = [int("20"+val.split("'")[1]) if type(val)==str else val for val in qtrs ]
cpu_df['launch_year'] = qtrs
prices = cpu_df['Recommended_Customer_Price'].tolist()
filtered_prices = []

for price in prices:

    if type(price) == str:

        if '-' in price:

            p1 = float(price.split('-')[0].strip()[1:])

            p2 = float(price.split('-')[1].strip()[1:])

            avg = (p1+p2)/2

            filtered_prices.append(avg)

        else:

            filtered_prices.append(price[1:])

    else:

        filtered_prices.append(price)
cpu_df['rec_cust_prices'] = filtered_prices
cpu_df['rec_cust_prices']=cpu_df['rec_cust_prices'].replace("[,]",'',regex=True).astype(float)
import matplotlib.pyplot as plt
plt.scatter(cpu_df['launch_year'], cpu_df['rec_cust_prices'])

plt.show()
cp