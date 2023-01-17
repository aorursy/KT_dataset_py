# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics as st

import matplotlib.pyplot as plt 

import json 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans



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

#     data1=data[0]

#     print(type(data1['id']))

#     print(type(data1['list']))

    # data1

    data2=[]

    gravity=[]

    rotation=[]

    attitude=[]

    userAcceleration=[]

    time=[]

    for i in range (0, len(data)):

        data1 = data[i]

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



#     json_final = pd.concat([Time, attitude_x, attitude_y, attitude_z, 

#                     rotation_roll, rotation_pitch, rotation_yaw, 

#                     gravity_x, gravity_y, gravity_z, 

#                     userAcceleration_x, userAcceleration_y, userAcceleration_z], 

#                    axis=1, sort=False)



    json_final = pd.concat([Time, rotation_roll, rotation_pitch, rotation_yaw,

                            attitude_x, attitude_y, attitude_z, 

                            gravity_x, gravity_y, gravity_z, 

                            userAcceleration_x, userAcceleration_y, userAcceleration_z], 

                           axis=1, sort=False)

#     json_final = pd.concat([Time, attitude_x, attitude_y, attitude_z, 

#                              userAcceleration_x, userAcceleration_y, userAcceleration_z,

#                              gravity_x, gravity_y, gravity_z, 

#                              rotation_roll, rotation_pitch, rotation_yaw], 

#                             axis=1, sort=False)

    json_final



    return json_final

# len(gravity)

# data3[0]

# userAccelerationx[1]
def convert_data(pd_data):

    pdd = pd_data.loc[:,:].values

    return(pdd)
def cluster_data(d):

    kmeans = KMeans(n_clusters=5, max_iter=500)

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
def join_array(d3, dnew):

    return(np.concatenate((d3, dnew), axis=0))
#Import Gaussian Naive Bayes model

def predict_naive(x_train, y, x_test):

    from sklearn.naive_bayes import GaussianNB



    #Create a Gaussian Classifier

    model = GaussianNB()



    # Train the model using the training sets

    model.fit(x_train,y)



#     to_pred = data9_pd.loc[:,:].values

#     to_pred = to_pred[:,1:]



    #Predict Output

    predicted= model.predict(x_test) # 0:Overcast, 2:Mild

#     print("Predicted Value:", predicted)

    return(predicted)
# Training Data

# tbo = '/kaggle/input/events_export 3.json'

# tbo = files_json[1]

tbo = files_json[2] # was 0 instead of 2

# datapd = pd.read_json(tbo)

# datapd

train_json = open_json(tbo)

train_data = explore_json(train_json)



traincv_data = convert_data(train_data)

traincv_data = traincv_data[:,1:]

train_data.shape
# for i in range (1, len(files_json)):

train1_json = open_json(files_json[8]) # was 2 instead of 8

train1_data = explore_json(train1_json)

newdd = join_array(train_data, train1_data)



# train1_json = open_json(files_json[4])

# train1_data = explore_json(train1_json)

# newdd = join_array(newdd, train1_data)



# train1_json = open_json(files_json[5])

# train1_data = explore_json(train1_json)

# newdd = join_array(newdd, train1_data)



# train1_json = open_json(files_json[6])

# train1_data = explore_json(train1_json)

# newdd = join_array(newdd, train1_data)



train1_json = open_json(files_json[1])

train1_data = explore_json(train1_json)

newdd = join_array(newdd, train1_data)



train1_json = open_json(files_json[9])

train1_data = explore_json(train1_json)

newdd = join_array(newdd, train1_data)



# train1_json = open_json(files_json[9])

# train1_data = explore_json(train1_json)

# newdd = join_array(newdd, train1_data)





newdd



#traincv_data = convert_data(newdd)

# traincv_data = newdd[:,1:]

traincv_data = traincv_data[:,:]



traincv_data.shape
def apply_pca(da):

    from sklearn.decomposition import PCA 



    pca = PCA(n_components = 1) 



    X_1 = pca.fit_transform(da) 

    # X_2 = pca.transform(X_test) 



    # explained_variance = pca.explained_variance_ratio_ 

    X_1.shape

    return(X_1)

traincv_data.shape

av = np.zeros((traincv_data.shape[0],1))

for i in range (1,traincv_data.shape[0]):

    av[i] = st.mean(traincv_data[i,:])



# av

plt.plot(av[:,:])

def find_vars(X_1):

    wa = np.zeros((X_1.shape[0],1))

    for i in range (1,X_1.shape[0]):

        wa[i] = st.mean(X_1[i,:])



    plt.plot(wa)

    vvr = np.zeros((wa.shape[0],1))

    for i in range (0, wa.shape[0]-15):

        vvr[i] = st.variance(wa[i:i+14,0])

    return(vvr)

plt.plot(vvr)
X_pca = apply_pca(traincv_data)

vvr = find_vars(X_pca)
#Clustering data

mydata = convert_data(train_data)

mydata = mydata[:,1:]

# labels, centroids = cluster_data(mydata)

# traincv1_data = traincv_data[700+116:,:]

# labels, centroids = cluster_data(traincv_data)

labels, centroids = cluster_data(vvr)

training_data = export_training_data(mydata, labels, centroids)

training_data.to_csv(r'/kaggle/working/training_data.csv', index = False)

labels

plt.plot(labels)

# plt.plot(vvr)
plt.plot(vvr)
vvr1 = vvr.tolist()

mop = vvr1.index(max(vvr))

mop

labels[mop]



mip = vvr1.index(min(vvr))

labels[mip]

# Testing Data

# tbo = '/kaggle/input/events_export 2.json'

tbo = files_json[2]

test_json = open_json(tbo)

test_data = explore_json(test_json)

# test_data.shape

X_t = test_data.loc[:,:].values

X_test = X_t[:,1:]

X_test.shape
X_pca_testing = apply_pca(X_test)

vvr_testing = find_vars(X_pca_testing)
labels.shape
#Defining training and testing sets

# X_training = vvr

# Y_training = labels

# X_testing = vvr_testing



X_training = vvr[0:1000]

Y_training = labels[0:1000]

X_testing = vvr_testing[996:]
X_pca.shape

# vvr.shape

# vvr_testing.shape
# Training & Prediction of the model

y_pred = predict_model(X_training, X_testing, Y_training)

y_pred

plt.plot(y_pred)
Y_training.shape

y_pred.shape
# print("Number of mislabeled points out of a total %d points : %d", (vvr.shape[0], (labels != y_pred).sum()))

same = (Y_training != y_pred).sum()

# print('It is ', )

print("Number of mislabeled points out of a total %d points : %d", (y_pred.shape, (Y_training != y_pred).sum()))

print("It is ", (100-(Y_training != y_pred).sum()/y_pred.shape*100), "percent accurate")