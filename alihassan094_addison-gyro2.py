# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics as st

import matplotlib.pyplot as plt 

import json 

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
files=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file1 = os.path.join(dirname, filename)

        files.append(file1)

files
files_json = [f for f in files if f[-4:] == 'json']

print('There are total ', len(files_json), 'files')
files_json.sort()

files_json

# events_export 2 = 10 for start to mouth

# events_export 3 = 10 for full

# events_export 4 = 10 while sitting
df = pd.DataFrame()

shp = np.zeros(80)

i=0



for f in files_json:

    

    data_all = pd.read_json(f)

    shp[i] = len(data_all)

    df = df.append(data_all, sort=False)

    i = i+1

    

 

    

print(df.shape)

df.to_csv(r'/kaggle/working/file1.csv', index = False)

# df.drop(['f1'])

# print(df.loc[810].values)

df

def open_json(myurl):



    # Opening JSON file 

    f = open(myurl,) 



    # returns JSON object as  

    # a dictionary 

    data = json.load(f) 



    # Closing file 

    f.close() 

    return data

# len(data)
def explore_json(data):

#     print(type(data[0]))

#     print(type(data[1]))

#     print(type(data[2]))

#     print(type(data[3]))

    # print(type(data[4]))

    data1=data[0]

#     print(type(data1['id']))

#     print(type(data1['list']))

    # data1

    data2=[]

    gravity=[]

    rotation=[]

    attitude=[]

    userAcceleration=[]

    time=[]

    for person in data1['list']:

    #     temp=person

        temg=person['gravity']

    #     print(temp)

        gravity.append(temg)

        temr=person['rotation']

        rotation.append(temr)

        tema=person['attitude']

        attitude.append(tema)

        temu=person['userAcceleration']

        userAcceleration.append(temu)

        temt=person['time']

        time.append(temt)





    gravityx=[]

    gravityy=[]

    gravityz=[]

    rotationroll=[]

    rotationyaw=[]

    rotationpitch=[]

    attitudex=[]

    attitudey=[]

    attitudez=[]

    userAccelerationx=[]

    userAccelerationy=[]

    userAccelerationz=[]



    for i in range (len(gravity)):

        temgx = gravity[i]['x']

        gravityx.append(temgx)

        temgy = gravity[i]['y']

        gravityy.append(temgy)

        temgz = gravity[i]['z']

        gravityz.append(temgz)



        temrroll = rotation[i]['roll']

        rotationroll.append(temrroll)

        temryaw = rotation[i]['yaw']

        rotationyaw.append(temryaw)

        temrpitch = rotation[i]['pitch']

        rotationpitch.append(temrpitch)



        temax = attitude[i]['x']

        attitudex.append(temax)

        temay = attitude[i]['y']

        attitudey.append(temay)

        temaz = attitude[i]['z']

        attitudez.append(temaz)



        temux = userAcceleration[i]['x']

        userAccelerationx.append(temux)

        temuy = userAcceleration[i]['y']

        userAccelerationy.append(temuy)

        temuz = userAcceleration[i]['z']

        userAccelerationz.append(temuz)

        

    Time = pd.DataFrame(time)

    gravity_x = pd.DataFrame(gravityx)

    gravity_y = pd.DataFrame(gravityy)

    gravity_z = pd.DataFrame(gravityz)



    rotation_roll = pd.DataFrame(rotationroll)

    rotation_pitch = pd.DataFrame(rotationpitch)

    rotation_yaw = pd.DataFrame(rotationyaw)



    attitude_x = pd.DataFrame(attitudex)

    attitude_y = pd.DataFrame(attitudey)

    attitude_z = pd.DataFrame(attitudez)



    userAcceleration_x = pd.DataFrame(userAccelerationx)

    userAcceleration_y = pd.DataFrame(userAccelerationy)

    userAcceleration_z = pd.DataFrame(userAccelerationz)



    json_final = pd.concat([Time, attitude_x, attitude_y, attitude_z, 

                    rotation_roll, rotation_pitch, rotation_yaw, 

                    gravity_x, gravity_y, gravity_z, 

                    userAcceleration_x, userAcceleration_y, userAcceleration_z], 

                   axis=1, sort=False)



    json_final1 = pd.concat([Time, rotation_roll, rotation_pitch, rotation_yaw,

                            attitude_x, attitude_y, attitude_z, 

                            gravity_x, gravity_y, gravity_z, 

                            userAcceleration_x, userAcceleration_y, userAcceleration_z], 

                           axis=1, sort=False)

    json_final



    return json_final

# len(gravity)

# data3[0]

# userAccelerationx[1]
def convert_data(pd_data):

    pdd = pd_data.loc[:,:].values

    return(pdd)
def cluster_data(d):

    kmeans = KMeans(n_clusters=2, max_iter=300)

    kmeans.fit(d)

    labels = kmeans.predict(d)

    centroids = kmeans.cluster_centers_

    return(labels, centroids)

def export_training_data(d, lb, cn):

    d1 = pd.DataFrame(d)

    lb1 = pd.DataFrame(lb)

    exporting_data = pd.concat([d1, lb1], axis=1, sort=False) 

    return(exporting_data)
#Train the model

def predict_model(X_train, X_test, labels):

#     X_train = d[:,1:]

    y_train = labels

    classifier = KNeighborsClassifier(n_neighbors=5)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    return y_pred
# Training Data

# tbo = '/kaggle/input/events_export 3.json'

tbo = files_json[1]

# datapd = pd.read_json(tbo)

# datapd

train_json = open_json(tbo)

train_data = explore_json(train_json)

train_data.shape

traincv_data = convert_data(train_data)

traincv_data = traincv_data[:,1:]

# type(train_data)
import matplotlib.pyplot as plt #for data visualization 

import seaborn as sns

plt.figure(1 , figsize = (17 , 12))



cor = sns.heatmap(train_data.corr(), annot = True)
num=[]

for i in range (traincv_data.shape[0]):

    num.append(i)

    i=i=1

num
plt.scatter(num, traincv_data[:,2], label= "stars", color= "green", marker= "*", s=30)
# Finding the variance to know how much the data is varying

var=[]

for i in range (1, traincv_data.shape[1]):

    var1 = st.variance(traincv_data[:,i])

    var.append(var1)

plt.plot(var)

len(var)

# Just trying filtering of data

d1 = pd.DataFrame({

    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],

    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]

})



from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3)

kmeans.fit(d1)
#Clustering data

mydata = convert_data(train_data)

mydata = mydata[:,1:]

labels, centroids = cluster_data(mydata)

training_data = export_training_data(mydata, labels, centroids)

training_data.to_csv(r'/kaggle/working/training_data.csv', index = False)
# Testing Data

# tbo = '/kaggle/input/events_export 2.json'

tbo = files_json[4]

test_json = open_json(tbo)

test_data = explore_json(test_json)

# test_data.shape

X_t = test_data.loc[:,:].values

X_test = X_t[:,1:]

X_test.shape
# Training & Prediction of the model

y_pred = predict_model(traincv_data, X_test, labels)

y_pred
labels

if labels[1] == 0:

    print('0 means glass is in lifting position')

    pl=labels[1]

elif (labels[1]==1):

    print('1 means glass is in lifting position')

    pl=labels[1]

elif (labels[1]==2):

    print('2 means glass is in lifting position')

    pl=labels[1]

else:

    print('Try Again')



# if (labels[len(labels)-1] == 0):

#     print('0 means glass is in retrieving position')

#     rt=labels[len(labels)-1]

# elif (labels[len(labels)-1]==1):

#     print('1 means glass is in retrieving position')

#     rt=labels[len(labels)-1]

# elif (labels[len(labels)-1]==2):

#     print('2 means glass is in retrieving position')

#     rt=labels[len(labels)-1]

# else:

#     print('Try Again')



if (st.median(labels) == 0):

    print('0 means glass is in mid position')

    rt=st.median(labels)

elif (st.median(labels)==1):

    print('1 means glass is in mid position')

    rt=st.median(labels)

elif (st.median(labels)==2):

    print('2 means glass is in mid position')

    rt=st.median(labels)

else:

    print('Try Again')



pred=np.array(len(y_pred))

y_pred.shape

type(y_pred)

y_pred[43]



pred=[]

for c in range (0,len(y_pred)):

    if y_pred[c]==pl:

        a='Picking'

        pred.append(a)

#         print('It is in picking position')

    elif y_pred[c]==rt:

        a='At Mid'

        pred.append(a)

#         print('It is in retrieving position')

    

X_test.shape

len(pred)

predd=np.array(pred)

predd.shape

pn = pd.DataFrame(predd)

pn
# finale = np.concatenate((X_test, predd), axis=1)

Xt1 = pd.DataFrame(X_test)

predd = pd.DataFrame(pred)

predd

Xt1

finale = pd.concat([test_data, predd], axis=1, sort=False)

finale

# train_data = pd.DataFrame(kk)



# finale.to_csv(r'/kaggle/working/Finale.csv', index = False)



# finale.to_csv(r'/kaggle/working/Result of events_export 2.csv', index = False)

# finale.to_csv(r'/kaggle/working/Result of events_export 6.csv', index = False)

finale.to_csv(r'/kaggle/working/Result of events_export 7.csv', index = False)