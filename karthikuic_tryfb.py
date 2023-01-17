 # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import os
import numpy as np
import pandas as pd
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from bokeh.plotting import figure, output_file, show
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

# Any results you write to the current directory are saved as output.
print('done')
train_dir = "../input"
train_file = "train.csv"
fbcheckin_train_tbl = pd.read_csv(os.path.join(train_dir, train_file))


# Assuming time values were time elapsed epoch times. we can now calculate the hours, minutes, days and seconds from last checkin 

fbcheckin_train_tbl['timeinmin'] = fbcheckin_train_tbl['time']
fbcheckin_train_tbl['time_of_week'] = fbcheckin_train_tbl['time'] % 10080
fbcheckin_train_tbl['hour_of_day']  = (fbcheckin_train_tbl['time_of_week'] / 60) % 24
fbcheckin_train_tbl['hour_number_for_week'] = fbcheckin_train_tbl['time'] % (10080) //60.
fbcheckin_train_tbl['day_of_week'] = fbcheckin_train_tbl['hour_number_for_week'] // 24.



#one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]['timeinmin'] % (10080) //60.
#n, bins, patches = plt.hist(one_id, 168 , histtype='bar')
#plt.show()

print(fbcheckin_train_tbl.head(10))

one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]['timeinmin'] % (10080) //60.
n, bins, patches = plt.hist(one_id, 168 , histtype='bar')
plt.show()
