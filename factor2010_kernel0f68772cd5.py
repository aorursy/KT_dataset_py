# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from mpl_toolkits import mplot3d

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/indoor-navigation-system/sensors.csv")

data.head()
print('Shape of the data set: ' + str(data.shape))

# row and coloumn
Temp = pd.DataFrame(data.isnull().sum())

Temp.columns = ['Sum']

print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])) )
plt.title('Accelerometer')

plt.xlabel('Time(s)')

plt.ylabel('Acceleration')

aX=data['Accelerometer X (g)']

aY=data['Accelerometer Y (g)']

aZ=data['Accelerometer Z (g)']

Time=data['Time (s)']

plt.plot(aX, label = "X", color='red')

plt.plot(aY,label = "Y", color='blue')

plt.plot(aZ,label = "Z", color='green')

plt.legend() 





plt.title('Gyroscope')

plt.xlabel('Time(s)')

plt.ylabel('Angylar velocity')

gX=data['Gyroscope X (deg/s)']

gY=data['Gyroscope Y (deg/s)']

gZ=data['Gyroscope Z (deg/s)']

Time=data['Time (s)']

plt.plot(gX, label = "X", color='red' )

plt.plot(gY, label = "Y", color='blue')

plt.plot(gZ, label = "Z", color='green') 

plt.legend() 



plt.title('Magnetometer')

plt.xlabel('Time(s)')

plt.ylabel('Magnetometer')

MX=data['Magnetometer X (uT)']

MY=data['Magnetometer Y (uT)']

MZ=data['Magnetometer Z (uT)']

Time=data['Time (s)']

plt.plot(MX, label = "X", color='red' )

plt.plot(MY, label = "Y", color='blue')

plt.plot(MZ, label = "Z", color='green') 

plt.legend() 
plt.title('Velocity')

plt.xlabel('Time(s)')

plt.ylabel('Velocity (m\s)')

VX=data['VelocityX']

VY=data['VelocityY']

VZ=data['VelocityZ']

Time=data['Time (s)']

plt.plot(VX, label = "X", color='red' )

plt.plot(VY, label = "Y", color='blue')

plt.plot(VZ, label = "Z", color='green') 

plt.legend() 
plt.title('Position')

plt.xlabel('Time(s)')

plt.ylabel('Position (m)')

PX=data['PositionX']

PY=data['PositionY']

PZ=data['PositionZ']

Time=data['Time (s)']

#plt.plot(PX, label = "X", color='red' )

#plt.plot(PY, label = "Y", color='blue')

#plt.plot(PZ, label = "Z", color='green' ) 

plt.legend() 
plt.title('Acceleration Magnitude')

plt.xlabel('Time(s)')

plt.ylabel('Magnitude (g)')

AmagX=data['Accelerometer_Magnitude']

plt.plot(AmagX, label = "X", color='red' )

plt.title('Linear Accleration')

plt.xlabel('Time(s)')

plt.ylabel('Linear Acceleration (g)')

LaccX=data['LinearAccelerationX (g)']

LaccY=data['LinearAccelerationY (g)']

LaccZ=data['LinearAccelerationZ (g)']

Time=data['Time (s)']

plt.plot(LaccX, label = "X", color='red' )

plt.plot(LaccY, label = "Y", color='blue')

plt.plot(LaccZ, label = "Z", color='green' ) 

plt.legend() 

plt.title('Euler Angle')

plt.xlabel('Time(s)')

plt.ylabel('Angle')



Pitch=data['Pitch (deg)']

Yaw=data['Yaw (deg)']

Roll=data['Roll (deg)']

#plt.plot(Roll, label = "Roll", color='red' )

#plt.plot(Pitch, label = "Pitch", color='blue')

#plt.plot(Yaw, label = "Yaw", color='green' ) 

plt.legend() 















#im = plt.imread("../input/floorplan/Floorplan.png")

#imgplot = plt.imshow(im)





data1 = pd.read_csv("../input/alphabetafilter/Learning.csv")

data.head()

plt.title('Pridect Position using Alpha-Beta Filter Algorithm')

plt.xlabel('Time(s)')

plt.ylabel('Position (m)')



Xalpha=data1['Xalpha']

Yalpha=data1['Yalpha']

Zalpha=data1['Zalpha']

plt.plot(Xalpha, label = "X", color='red' )

plt.plot(Yalpha, label = "y", color='blue')

plt.plot(Zalpha, label = "z", color='green' ) 

plt.legend() 
plt.title('Pridect Position using Learning to Alpha-Beta Filter Algorithm')



plt.xlabel('Time(s)')

plt.ylabel('Position (m)')

xpos=data1['X']

ypos=data1['Y']

zpos=data1['Z']

LalpX=data1['XLeanAlpha']

LalphaY=data1['Ylearn']

LalphaZ=data1['Zlearn']



plt.plot(xpos, label = "pos(x)", color='orange' )

plt.plot(ypos, label = "pos(y)", color='navy' )



plt.plot(zpos, label = "pos(z)", color='lime' )

plt.plot(Xalpha, label = "α,β pos(z)" , color='purple' )



plt.plot(Yalpha, label = "α,β pos(z)", color='fuchsia' )



plt.plot(Zalpha, label = "α,β pos(z)",  color='teal')





plt.plot(LalpX, label = "Learning-α,β(X)", color='red' )

plt.plot(LalphaY, label = "Learning-α,β(Y)", color='blue')

plt.plot(LalphaZ, label = "Learning-α,β(Z)", color='green' ) 

plt.legend() 
fig = plt.figure()

fig = plt.figure(figsize=(15, 15))

ax = plt.axes(projection='3d')





ax.plot3D(xpos, ypos, zpos)

ax.plot3D(Xalpha, Yalpha, Zalpha)

ax.plot3D(LalpX, LalphaY, LalphaZ)

ax.set_xlabel('Postion x')

ax.set_ylabel('position y')

ax.set_zlabel('Position z');


fig = plt.figure()

fig = plt.figure(figsize=(15, 15))

ax = plt.axes(projection='3d')



xline = data['PositionX']

yline = data['PositionY']

zline = data['PositionZ']

ax.plot3D(xline, yline, zline,marker='o', color='red')

ax.set_xlabel('Postion x')

ax.set_ylabel('position y')

ax.set_zlabel('Position z');


position_data = data[['PositionX','PositionY','PositionZ','Time (s)']]

position_data.head()



position_data = position_data[np.isfinite(position_data['PositionX'])]
pd.isnull(position_data.PositionX).sum()
position_data = position_data[np.isfinite(position_data['PositionY'])]
pd.isnull(position_data.PositionY).sum()
position_data = position_data[np.isfinite(position_data['PositionZ'])]
pd.isnull(position_data.PositionZ).sum()
position_data.dropna(inplace=True)

position_data.drop_duplicates(inplace=True)

sns.pairplot(position_data);