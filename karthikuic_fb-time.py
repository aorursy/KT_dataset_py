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
print (fbcheckin_train_tbl.head(10))


# Assuming time values were time elapsed epoch times. we can now calculate the hours, minutes, days and seconds from last checkin 

fbcheckin_train_tbl['timeinmin'] = fbcheckin_train_tbl['time']
fbcheckin_train_tbl['time_of_week'] = fbcheckin_train_tbl['time'] % 10080
fbcheckin_train_tbl['hour_of_day']  = (fbcheckin_train_tbl['time_of_week'] / 60) % 24
fbcheckin_train_tbl['hour_number_for_week'] = fbcheckin_train_tbl['time'] % (10080) //60.
fbcheckin_train_tbl['day_of_week'] = fbcheckin_train_tbl['hour_number_for_week'] // 24.




one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]['hour_number_for_week']
n, bins, patches = plt.hist(one_id, 168 , histtype='bar')
plt.show()



                                  








# check for cycles over the course of a day (top) and week (bottom)
def plotMeanbyday(selected_id):
    f, axarr = plt.subplots(3,3,figsize=(15, 9));
    for j in range(0,7):
        fd_by_day = selected_id[selected_id['day_of_week'] == j]['hour_number_for_week'] % 24
        #print('Day',j,'Median',int(fd_by_day.median()))
        #n, bins, patches = axs[j].hist(fd_by_day,7);
        
        if j == 0:
            axarr[0,0].hist(fd_by_day,7);
            axarr[0,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,0].set_title('Day 1')
        if j == 1:
            axarr[0,1].hist(fd_by_day,7);
            axarr[0,1].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,1].set_title('Day 2')
        if j == 2:
            axarr[0,2].hist(fd_by_day,7);
            axarr[0,2].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,2].set_title('Day 3')
        if j == 3:
            axarr[1,0].hist(fd_by_day,7);
            axarr[1,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,0].set_title('Day 3')
        if j == 4:
            axarr[1,1].hist(fd_by_day,7);
            axarr[1,1].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,1].set_title('Day 4')
        if j == 5:
            axarr[1,2].hist(fd_by_day,7);
            axarr[1,2].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,2].set_title('Day 5')
        if j == 6:
            axarr[2,0].hist(fd_by_day,7);
            axarr[2,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[2,0].set_title('Day 6')
        
            
        plt.xlabel('Time (Hours)');
        plt.ylabel('Frequency');
        f.show();
        
selected_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]   
plotMeanbyday(selected_id)
fd_by_day_allweeks = selected_id['hour_number_for_week'] % 24
fd_by_day_allweeksmedian = fd_by_day_allweeks.median()
print('Place_id_median',fd_by_day_allweeksmedian) 
print('Place_id_mean',int(fd_by_day_allweeks.mean())) 
# check for cycles over the course of a day (top) and week (bottom)
def plotMeanbyday(selected_id):
    f, axarr = plt.subplots(3,3,figsize=(15, 9));
    for j in range(0,7):
        fd_by_day = selected_id[selected_id['day_of_week'] == j]['hour_number_for_week'] % 24
        #print('Day',j,'Median',int(fd_by_day.median()))
        #n, bins, patches = axs[j].hist(fd_by_day,7);
        
        if j == 0:
            axarr[0,0].hist(fd_by_day,7);
            axarr[0,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,0].set_title('Day 1')
        if j == 1:
            axarr[0,1].hist(fd_by_day,7);
            axarr[0,1].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,1].set_title('Day 2')
        if j == 2:
            axarr[0,2].hist(fd_by_day,7);
            axarr[0,2].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[0,2].set_title('Day 3')
        if j == 3:
            axarr[1,0].hist(fd_by_day,7);
            axarr[1,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,0].set_title('Day 3')
        if j == 4:
            axarr[1,1].hist(fd_by_day,7);
            axarr[1,1].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,1].set_title('Day 4')
        if j == 5:
            axarr[1,2].hist(fd_by_day,7);
            axarr[1,2].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[1,2].set_title('Day 5')
        if j == 6:
            axarr[2,0].hist(fd_by_day,7);
            axarr[2,0].axvline(fd_by_day.mean(), color='r', linestyle='dashed', linewidth=2)
            axarr[2,0].set_title('Day 6')
        
            
        plt.xlabel('Time (Hours)');
        plt.ylabel('Frequency');
        f.show();
        
selected_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==1308450003]   
plotMeanbyday(selected_id)
fd_by_day_allweeks = selected_id['hour_number_for_week'] % 24
fd_by_day_allweeksmedian = fd_by_day_allweeks.median()
print('Place_id_median',fd_by_day_allweeksmedian) 
print('Place_id_mean',int(fd_by_day_allweeks.mean())) 
fbcheckin_train_tbl['hour_dist'] = fbcheckin_train_tbl['hour_number_for_week'] % 24 
one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]
filtered = fbcheckin_train_tbl[['x','y','hour_dist']]

### And the same exploration by weekday
### based on localization info
ids = ['8523065625','1623394281']
fig = plt.figure(figsize=(15,15))

for day in range(7):
    sp = fig.add_subplot(3,3,day+1)
    x = one_id[one_id['day_of_week'] == day]['x']
    y = one_id[one_id['day_of_week'] == day]['y']
    sp.hist2d(x, y, bins=20, norm=LogNorm())
    sp.set_title('x and y location histogram day %d' % day)
    sp.set_xlabel('x')
    sp.set_ylabel('y')
plt.show()
# seems the outliers are hugley skeyed along the x axis 
fbcheckin_train_tbl['hour_dist'] = fbcheckin_train_tbl['hour_number_for_week'] % 24 
one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==1308450003]
filtered = fbcheckin_train_tbl[['x','y','hour_dist']]

### And the same exploration by weekday
### based on localization info
ids = ['8523065625','1623394281']
fig = plt.figure(figsize=(10,10))

for day in range(7):
    sp = fig.add_subplot(3,3,day+1)
    x = one_id[one_id['day_of_week'] == day]['x']
    y = one_id[one_id['day_of_week'] == day]['y']
    sp.hist2d(x, y, bins=20, norm=LogNorm())
    sp.set_title('x and y location histogram day %d' % day)
    sp.set_xlabel('x')
    sp.set_ylabel('y')
plt.show()


# the deviation seems to be consistantly on the x axis ! the spread on x is huge ??
one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==8523065625]
x_centre = one_id['x'].sum() / len(one_id['x'])
y_centre = one_id['y'].sum() / len(one_id['y'])
one_id['deviation_x'] = one_id['x'] - x_centre
one_id['deviation_y'] = one_id['y'] - y_centre
#print (one_id.head(10))
# red dashes, blue squares and green triangles
#plt.plot(one_id['deviation_y'],one_id['accuracy'],'.')
#plt.show()
fig = plt.figure(figsize=(10,10))
ids = ['8523065625','1623394281','1308450003','6289802927','9931249544','5662813655','1137537235']


for id in ids:
    sp = fig.add_subplot(3,3,int(ids.index(id)+1))
    one_id = fbcheckin_train_tbl[fbcheckin_train_tbl['place_id']==int(id)]
    x_centre = one_id['x'].sum() / len(one_id['x'])
    y_centre = one_id['y'].sum() / len(one_id['y'])
    one_id['deviation_x'] = one_id['x'] - x_centre
    one_id['deviation_y'] = one_id['y'] - y_centre
    sp.plot(one_id['deviation_x'],one_id['accuracy'],'.')
    sp.set_title('x-deviation and accuracy')
    sp.set_xlabel('x')
    sp.set_ylabel('accuracy')
plt.show()
# From the deviations of x from the centroid and plotting the accuracy we can now see that the accuracy
# goes beserk as the checkin is more near to the location. Or the GPS goes crazy when the customer is 
# inside the building. 
# Also look at the spread. Some are more clustered around the 0s. Why is that. This means that the
# accuracys are more often to be inaccurate. May be these places are near high rise buildings. Why so ? 
# as high rise buildings block sky visibiity. So does visibility to GPS Satellites. So some locations are 
# near downtown may be ? or surrounded by sky rise buildings. 

