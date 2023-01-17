# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Client Cancellations0.csv")

df.Days = [each if each >= 0 else 0.0 for each in df.Days]

plt.hist(df.Days, bins=60)
plt.xlabel("Day difference between cancellation and service dates")
plt.ylabel("Cancellation frequency by day difference")
plt.title("Day difference and cancellation frequency relation")
plt.show()
data = pd.read_csv("../input/hair_salon_no_show_wrangled_df.csv")

# correlation map
f, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)
plt.show()
data.last_cumnoshow[0:500].plot(color='red', figsize=(30, 10))
data.last_cumbook[0:500].plot(color='green', figsize=(30, 10))
data.last_cumcancel[0:500].plot(color='blue', figsize=(30, 10))
plt.show()
data.last_cumnoshow[500:1000].plot(color='red', figsize=(30, 10))
data.last_cumbook[500:1000].plot(color='green', figsize=(30, 10))
data.last_cumcancel[500:1000].plot(color='blue', figsize=(30, 10))
plt.show()
data.last_cumnoshow[1000:1500].plot(color='red', figsize=(30, 10))
data.last_cumbook[1000:1500].plot(color='green', figsize=(30, 10))
data.last_cumcancel[1000:1500].plot(color='blue', figsize=(30, 10))
plt.show()
data.last_cumnoshow[1500:2000].plot(color='red', figsize=(30, 10))
data.last_cumbook[1500:2000].plot(color='green', figsize=(30, 10))
data.last_cumcancel[1500:2000].plot(color='blue', figsize=(30, 10))
plt.show()
noshowfilter = data.last_cumnoshow > 0
noshowstaffs = data.book_staff[(data.last_cumnoshow[noshowfilter])]

noshowstaffs = [1 if each == 'JJ' else each for each in noshowstaffs]
noshowstaffs = [2 if each == 'JOANNE' else each for each in noshowstaffs]
noshowstaffs = [3 if each == 'KELLY' else each for each in noshowstaffs]
noshowstaffs = [4 if each == 'BECKY' else each for each in noshowstaffs]

print('Total no-show: ', len(noshowstaffs))
print('Total booking: ', len(data.book_staff))

plt.hist(noshowstaffs, bins=4, color='red')
plt.title('staff-noshow count relation')
plt.ylabel('no-show count')
plt.xlabel('staff id:{JJ: 1, JOANNA: 2, KELLY: 3, BECKY: 4}')
plt.show()

data.book_staff = [1 if each == 'JJ' else each for each in data.book_staff]
data.book_staff = [2 if each == 'JOANNE' else each for each in data.book_staff]
data.book_staff = [3 if each == 'KELLY' else each for each in data.book_staff]
data.book_staff = [4 if each == 'BECKY' else each for each in data.book_staff]
data.book_staff = [5 if (each != 1 and each != 2 and each != 3 and each != 4) else each for each in data.book_staff]

plt.hist(data.book_staff, bins=5, color='blue')
plt.title('staff-reserve count relation')
plt.ylabel('reserve count')
plt.xlabel('staff id:{JJ: 1, JOANNA: 2, KELLY: 3, BECKY: 4, OTHERS: 5}')
plt.show()


