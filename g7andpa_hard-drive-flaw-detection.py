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
harddrive = pd.read_csv('../input/harddrive.csv')
print(harddrive.head())
print(harddrive.shape)
# check the number of failure.
a = harddrive[harddrive.failure == 1]
print(a.shape)
print(a.serial_number)
# Z300KHN0
keyValue = harddrive[harddrive.serial_number == 'Z300X1K0']
# print(Z302A13D)
pd.set_option('display.max_colwidth', 200)
keyValue[['date', 'smart_1_normalized', 'smart_5_raw', 'smart_7_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw', 'smart_240_raw']]
# keyValue[['smart_1_normalized', 'smart_5_raw', 'smart_7_raw', 'smart_187_raw', 'smart_197_raw']]

print(Z302A13D.smart_187_raw)
# see the numnber of disks.
all_series = harddrive.iloc[:, 1]
all_model = harddrive.iloc[:, 2]
num_disks = all_series.drop_duplicates()
# discard the duplicated disks
# num_disks = 65993
print(num_disks.shape)
# num_model = 69
num_model = all_model.drop_duplicates()
print(num_model.shape)
# 69 model
# 50 days' records with 65993 disks in 69 model types.
all_days = harddrive.iloc[:, 0]
num_days = all_days.drop_duplicates()
print(num_days.shape)

# find out all the records of the failure disks
# check if the 215 flaw disks are duplicated, the answer is yes there are 10 disks.
Serial_flaw_disks = a.iloc[:, 1].drop_duplicates()
# print(num_flaw_disks)
flaw_records = harddrive[harddrive.serial_number.isin(Serial_flaw_disks)]
print(flaw_records.shape)
# there are 5490 records that are related to the flaw disks.

# use the serial number and date to sort the data.
sort_flaw_records = flaw_records.sort_values(by=['serial_number', 'date'])
print(sort_flaw_records.smart_9_normalized)

# don't know how to deal with the data.
# I want to know what happen to the disks.
# WAS_data = sort_flaw_records[sort_flaw_records.serial_number=='13H883WAS']
flaw_data_5stats = sort_flaw_records.loc[:,['serial_number', 'smart_5_raw', 'smart_187_raw', 'smart_188_raw',
                                  'smart_189_raw' , 'smart_197_raw', 'smart_198_raw']]
# see what cause drives fail.

print(flaw_data_5stats.drop_duplicates('serial_number', keep='last'))
# show the 5 key stats that may cause drives fail
# print(flaw_data_5stats)
# smart_9_raw data means the hour that the drives have run. good feature.

# get feature data
data = harddrive.loc[:, [ 'serial_number', 'failure', 'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw',
                                  'smart_189_raw' , 'smart_197_raw', 'smart_198_raw']]
# y_data = harddrive.loc[:, 'failure']
print(data.head())
# print(y_data.head())
# wash the data, fillna with 0.
data_fina = data.fillna(value=0.0)
# print(X_data_fina.shape)=3179295, 7

#select the record that with one feature greater than one.
data_fina_b0 = data_fina[(data_fina.smart_5_raw>0) | (data_fina.smart_187_raw>0) | (data_fina.smart_188_raw>0) |
                            (data_fina.smart_189_raw>0) | (data_fina.smart_197_raw>0) | (data_fina.smart_198_raw>0)]
print(data_fina_b0.shape)
print(data_fina_b0.head())
import matplotlib.pyplot as plt
import seaborn as sbn
# 1: draw the failure records.0:draw the operational records.
fail_records = data_fina_b0[(data_fina_b0.failure==0)& (data_fina_b0.smart_188_raw<1400)& 
                            (data_fina_b0.smart_187_raw<0.5)& (data_fina_b0.smart_5_raw<1000)]
# fail_records = data_fina_b0[(data_fina_b0.failure==1) & (data_fina_b0.smart_5_raw<50000)& (data_fina_b0.smart_187_raw<50000)]

fail_records.plot.hexbin(x='smart_5_raw', y='smart_187_raw', gridsize=15)
# print(fail_records.head())

plt.title("5&187")
# plt.plot(data_fina_b0.smart_5_raw, data_fina_b0.smart_187_raw, '.', )
# plt.scatter(data_fina_b0.smart_5_raw, data_fina_b0.smart_187_raw, cmap=data_fina_b0.failure, marker='.')
# plt.scatter(fail_records.smart_188_raw, fail_records.smart_187_raw, marker='.')
# plt.show()
#draw 3d to see if there are some flaw information.
from mpl_toolkits.mplot3d import Axes3D
plt.figure(1)
ax1 = plt.subplot(221, projection='3d')
ax1.scatter(fail_records.smart_5_raw, fail_records.smart_187_raw, fail_records.smart_188_raw, marker='.')
ax1.set_title('5&187&188')

ax2 = plt.subplot(222, projection='3d')
ax2.scatter(fail_records.smart_5_raw, fail_records.smart_187_raw, fail_records.smart_189_raw, marker='.')
ax2.set_title('5&187&189')

ax3 = plt.subplot(223, projection='3d')
ax3.scatter(fail_records.smart_5_raw, fail_records.smart_187_raw, fail_records.smart_197_raw, marker='.')
ax3.set_title('5&187&197')

ax4 = plt.subplot(224, projection='3d')
ax4.scatter(fail_records.smart_5_raw, fail_records.smart_187_raw, fail_records.smart_198_raw, marker='.')
ax4.set_title('5&187&198')

plt.figure(2)
ax4 = plt.subplot(221, projection='3d')
ax4.scatter(fail_records.smart_5_raw, fail_records.smart_188_raw, fail_records.smart_189_raw, marker='.')
ax4.set_title('5&188&189')

ax4 = plt.subplot(222, projection='3d')
ax4.scatter(fail_records.smart_5_raw, fail_records.smart_188_raw, fail_records.smart_197_raw, marker='.')
ax4.set_title('5&188&197')

ax4 = plt.subplot(223, projection='3d')
ax4.scatter(fail_records.smart_5_raw, fail_records.smart_188_raw, fail_records.smart_198_raw, marker='.')
ax4.set_title('5&188&198')

ax4 = plt.subplot(224, projection='3d')
ax4.scatter(fail_records.smart_5_raw, fail_records.smart_189_raw, fail_records.smart_197_raw, marker='.')
ax4.set_title('5&189&197')


plt.show()
















