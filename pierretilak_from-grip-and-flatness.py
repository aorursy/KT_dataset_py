# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.plotting import figure, output_file, show

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/X_train.csv')

y = pd.read_csv('../input/y_train.csv')

test = pd.read_csv('../input/X_test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head(100)
y.head()
%matplotlib inline

import matplotlib.pyplot as plt



merged = pd.merge(y, train, on =['series_id'])



def plot_by_material(material):

    mat_y = y[y.surface==material]



    mat = pd.merge(concrete_y, train, on=['series_id'])



    #for i in mat.series_id:

    #    plt.plot(mat[mat.series_id==i].linear_acceleration_Z)

    #plt.show()

    

#plot_by_material('concrete')

#plot_by_material('hard tiles_large_space')

merged.head(10)
import math

def toEulerAngle(q):

    # roll (x-axis rotation)

    sinr_cosp = +2.0 * (q.w * q.x + q.y * q.z)

    cosr_cosp = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)

    roll = math.atan2(sinr_cosp, cosr_cosp)



    # pitch (y-axis rotation)

    sinp = +2.0 * (q.w * q.y - q.z * q.x)

    if (abs(sinp) >= 1):

        pitch = copysign(M_PI / 2, sinp) # use 90 degrees if out of range

    else:

        pitch = math.asin(sinp)



    #yaw (z-axis rotation)

    siny_cosp = +2.0 * (q.w * q.z + q.x * q.y)

    cosy_cosp = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll,pitch,yaw)
## Select a serie

m = merged[merged.series_id==12]



def quaternion2euler(m):

    ## Pass to Euler angles

    renamed_m = m.rename(columns={'orientation_X': 'x', 'orientation_Y': 'y', 'orientation_Z':'z', 'orientation_W':'w'})

    columns = ['x','y','z','w']

    euler = renamed_m[columns].apply(lambda x:toEulerAngle(x), axis=1)

    euler = euler.apply(pd.Series)

    euler.columns = ['roll','pitch','yaw']

    return euler

  

euler = quaternion2euler(m)

## Plot ROLL/PITCH/YAW

fig, axarr = plt.subplots(3, 1, figsize=(12, 8))

euler.roll.plot(ax=axarr[0])

euler.pitch.plot(ax=axarr[1])

euler.yaw.plot(ax=axarr[2])



#m.plot.scatter(x='orientation_X',y='orientation_Y',title='X/Y motions',ax=axarr[1])

col_acc = ['linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']

col_gyr = ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']



m[col_acc].plot()

m[col_gyr].plot()



def lpf(arr,alpha):

    old_val = 0

    result = []

    for i in arr:

        result.append(old_val*alpha + (1-alpha)*i)





euler.plot()
m.columns

from scipy.signal import butter, lfilter, medfilt,lfilter_zi

from scipy.signal import freqs



def low_pass_filter(data,order=1,alpha=0.05):

    b, a = butter(order,alpha)

    zi =  lfilter_zi(b, a)

    z,_ = lfilter(b, a,data,zi=zi*data.mean())

    return z





## Select a serie

m = merged[merged.series_id==12]

print(m.surface.values[0])



plt.figure() 

plt.plot(m['linear_acceleration_X'].values)

plt.plot(low_pass_filter(m['linear_acceleration_X'].values,alpha=0.1))

plt.figure() 

plt.plot(m['linear_acceleration_Y'].values)

plt.plot(low_pass_filter(m['linear_acceleration_Y'].values,alpha=0.1,order=1))

plt.figure() 

plt.plot(m['linear_acceleration_Z'].values)

plt.plot(low_pass_filter(m['linear_acceleration_Z'].values,alpha=0.001))





m = merged[merged.series_id==20]

print(m.surface.values[0])



plt.figure() 

plt.plot(m['linear_acceleration_Z'].values)

plt.plot(low_pass_filter(m['linear_acceleration_Z'].values,alpha=0.1))

## List of the different surfaces

surfaces = merged.surface.unique()



## Create a list of 1 serie by surface type

list_series = []



for s in surfaces:

    serie = merged[merged.surface==s].series_id.values[0]

    #print("look for %s, in serie %d"%(s,serie))

    list_series.append((serie,s))

for s in list_series:

    m = merged[merged.series_id==s[0]]

    #fig = plt.figure()

    #fig.suptitle(s[1])

    plt.plot(m['linear_acceleration_Z'].values,label=s[1])
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



## List of the different surfaces

surfaces = merged.surface.unique()



## Create a list of 1 serie by surface type

list_series = []



for s in surfaces:

    serie = merged[merged.surface==s].series_id.values[0]

    #print("look for %s, in serie %d"%(s,serie))

    list_series.append((serie,s))



scatter = []

for s in list_series:

    m = merged[merged.series_id==s[0]]

    y = m['linear_acceleration_Z'].values

    #y = low_pass_filter(m['linear_acceleration_Z'].values,alpha=0.01)

    scatter.append(go.Scatter(x=np.arange(128), y=y, mode='lines',name=s[1]))

   

iplot(scatter)
for s in list_series:

    m = merged[merged.series_id==s[0]]

    print(s[1])

    print(pd.Series(np.linalg.norm(m[col_acc].values,axis=1)).describe())
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



## List of the different surfaces

surfaces = merged.surface.unique()



## Create a list of 1 serie by surface type

list_series = []



for s in surfaces:

    serie = merged[merged.surface==s].series_id.values[0]

    #print("look for %s, in serie %d"%(s,serie))

    list_series.append((serie,s))



scatter = []

for s in list_series:

    m = merged[merged.series_id==s[0]]

    y = m['linear_acceleration_Z'].values

    #y = low_pass_filter(m['linear_acceleration_Z'].values,alpha=0.01)

    scatter.append(go.Scatter(x=np.arange(128), y=y, mode='lines',name=s[1]))

   

iplot(scatter)
from sklearn.model_selection import train_test_split



col_acc = ['linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']

col_gyr = ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z']



## List of series_id found in dataset

series = train.series_id.unique()



features_input = ['roll', 'pitch', 'norm_acc', 'norm_gyr']



def extract_features(m,serie_id):

    result = pd.DataFrame()

    for f in features_input:

        _std = m[f].std()

        _mean = m[f].mean()

        _min = m[f].min()

        _max = m[f].max()

        line = pd.Series([_std, _mean, _min, _max],

                index=[f+'_std',f+'_mean',f+'_min',f+'_max'])

        result = pd.concat([result,line],axis=0)

    #print(result)

    return pd.DataFrame(line)

        

df = pd.DataFrame()

for s in series[0:10]:

    ## We will work with m that contains the data of a specific serie id

    m = train[train.series_id==s]

    

    ## Process euler angles

    euler = quaternion2euler(m)

    ## Create norm for angular velocity and acceleration

    norm_gyr = pd.Series(np.linalg.norm(m[col_gyr].values,axis=1),name='norm_gyr')

    norm_acc = pd.Series(np.linalg.norm(m[col_acc].values,axis=1),name='norm_acc')

    

    m = pd.concat([m,euler,norm_acc,norm_gyr], axis=1)

    #print(m.columns)

    #print(m.describe())

    df.append(extract_features(m,s),ignore_index=True)

df