# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv")
data1=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv")

data2=pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv")

data3=pd.read_csv("../input/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv")
print("Bitstamp data shape : ", data1.shape)

print("Coinbase data shape : ", data2.shape)

print("BTCe data shape : ", data3.shape)
range1=(min(data1.loc[:,"Timestamp"]),max(data1.loc[:,"Timestamp"]))

range2=(min(data2.loc[:,"Timestamp"]),max(data2.loc[:,"Timestamp"]))



f, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.semilogy(data1.loc[:,"Timestamp"],data1.loc[:,"Weighted_Price"], color='r')

ax1.set_title('Bitstamp'); ax1.set_xlabel('time'); ax1.set_ylabel('price (USD)')

ax1.set_xlim([range1[0], range1[1]])



ax2.semilogy(data2.loc[:,"Timestamp"],data2.loc[:,"Weighted_Price"], color='b')

ax2.set_title('Coinbase'); ax2.set_xlabel('time'); 

ax2.set_xlim([range1[0], range1[1]])



ax3.semilogy(data3.loc[:,"Timestamp"],data3.loc[:,"Weighted Price"], color='g')

ax3.set_title('BTCe'); ax3.set_xlabel('time'); 

ax3.set_xlim([range1[0], range1[1]])