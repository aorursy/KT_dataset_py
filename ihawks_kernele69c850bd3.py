import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from seaborn import countplot,lineplot, barplot

import math

import lightgbm as lgb

import keras

from keras.models import Sequential

from keras.layers import Dense

from mlxtend.classifier import SoftmaxRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from numpy import argmax

from sklearn.metrics import accuracy_score

from keras.regularizers import l1

from scipy import stats

from keras.layers import Dropout

import statistics

import os

import random

print(os.listdir("../input"))
xtest = pd.read_csv("../input/X_test.csv")

xtrain = pd.read_csv("../input/X_train.csv")

ytrain = pd.read_csv("../input/y_train.csv")

ssb = pd.read_csv("../input/sample_submission.csv")
xtest.shape, xtrain.shape, ytrain.shape, ssb.shape
print('Number of surface: {}'.format(ytrain.surface.nunique()))

countplot(y = 'surface', data = ytrain)

plt.show()
xtrain.head()
ytrain.shape
xtest.head()
resultant_velocity = (xtrain['angular_velocity_X']**2+

xtrain['angular_velocity_Y']**2+  

xtrain['angular_velocity_Z']**2)**.5

resultant_velocity= pd.DataFrame(resultant_velocity, columns = ['resultant_velocity'])

#sc_x = StandardScaler()

#resultant_velocity_n = pd.DataFrame(sc_x.fit_transform(resultant_velocity),columns=resultant_velocity.columns, index=resultant_velocity.index )

#resultant_velocity_n = np.array(resultant_velocity_n)

resultant_velocity.shape

resultant_velocity.head()
resultant_acc = (xtrain['linear_acceleration_X']**2+

xtrain['linear_acceleration_Y']**2+  

xtrain['linear_acceleration_Z']**2)**.5

#resultant_acc= np.transpose(np.matrix(np.array(resultant_acc.T)))

resultant_acc= pd.DataFrame(resultant_acc, columns = ['resultant_acc'])

#sc_y = StandardScaler()

#resultant_acc = pd.DataFrame(sc_x.fit_transform(resultant_acc),columns=resultant_acc.columns, index=resultant_acc.index )

resultant_acc.head()
power=resultant_velocity['resultant_velocity']*resultant_acc['resultant_acc']

#As floor is fuction of friction, To estimate the friction factor differ force eq is required. Power is one of the factor on which friction force depends

power= pd.DataFrame(power, columns = ['power'])

power.head()

#xtrain_new = np.hstack((xtrain,resultant_velocity,resultant_acc,power))
#x, y, z, w = xtrain['orientation_X'].tolist(), xtrain['orientation_Y'].tolist(), xtrain['orientation_Z'].tolist(), xtrain['orientation_W'].tolist()

#t0 = 2*np.multiply(x,y)
#x = xtrain.iloc[:,3].values

#y = xtrain.iloc[:,4].values

#z = xtrain.iloc[:,5].values

#w = xtrain.iloc[:,6].values



#a0 = +2.0 * (np.multiply(w,x) + np.multiply(y , z))

#a1 = +1.0 - 2.0 * (np.multiply(x , x) + np.multiply(y , y))

#A = np.arctan2(a0,a1)



#a2 = +2.0 * (np.multiply(w , y) - np.multiply(z , x))

#B = np.arcsin(a2)



#a3 = +2.0 * (np.multiply(w , z) + np.multiply(x , y))

#a4 = +1.0 - 2.0 * (np.multiply(y , y) + np.multiply(z , z))

#C = np.arctan2(a3, a4)

#A.shape, B.shape, C.shape
#xtrain['euler_rotation_x'] = A

#xtrain['euler_rotation_y'] = B

#xtrain['euler_rotation_z'] = C

#xtrain['resultant_angle'] = (xtrain['euler_rotation_x'] ** 2 + xtrain['euler_rotation_y'] ** 2 + xtrain['euler_rotation_z'] ** 2)** .5
xtrain_new= pd.concat([xtrain,resultant_velocity, resultant_acc, power], axis=1)

xtrain_new.shape

xtrain_new.head()
merged = xtrain_new.merge(ytrain, on='series_id')

merged.shape
merged.head()
y =merged.iloc[:,16:18].values
#y = pd.DataFrame(y)

#y.head()
labelencoder = LabelEncoder()

y[:,1] = labelencoder.fit_transform(y[:, 1])

#y.shape

onehotencoder = OneHotEncoder(categorical_features = [1])

y = onehotencoder.fit_transform(y).toarray()

#y = onehotencoder.fit_transform(ytrain).toarray()

#Y_Train_f = Y_Train_f[:, 1:]
y_train = y[:,0:9]

y_train.shape

y_Train = pd.DataFrame(y_train)

y_Train.head()
y_train.shape
x = xtrain_new.iloc[:,3:16]

x.shape
X = pd.DataFrame(x)

X.head()
sc = StandardScaler()

x = sc.fit_transform(x)

x=pd.DataFrame(x)

x.head()
ytrain_new = xtrain_new.groupby('series_id')['power','resultant_velocity','resultant_acc'].mean()

ytrain_new = pd.DataFrame(ytrain_new).reset_index()

ytrain_new.columns = ['serie_id','avg_power','avg_velocity','avg_acc']

ytrain_new['surface'] = ytrain.surface

ytrain_new['group_id'] = ytrain.group_id
f, axes = plt.subplots(figsize=(10, 5))

sns.boxplot(x='surface',y='avg_power',data=ytrain_new, ax=axes)

plt.title('avg_power vs surface')
f, axes = plt.subplots(figsize=(10, 5))

sns.boxplot(x='surface',y='avg_acc',data=ytrain_new, ax=axes)

plt.title('avg_acc vs surface')
# Plot power vs velocity

f, axes = plt.subplots(figsize=(10, 5))

sns.lineplot(data=ytrain_new, x='avg_acc', y='avg_power', hue='surface',ax=axes)

plt.show()
f, axes = plt.subplots(figsize=(10, 5))

sns.lineplot(data=ytrain_new, x='avg_velocity', y='avg_power', hue='surface',ax=axes)
f, axes = plt.subplots(figsize=(10, 5))

sns.lineplot(data=ytrain_new, x='avg_velocity', y='avg_acc', hue='surface',ax=axes)
f, axes = plt.subplots(figsize=(10, 5))

sns.boxplot(x='surface',y='avg_power',data=ytrain_new, ax=axes)

plt.title('avg_power vs surface')
X_train, X_test, Y_train, Y_test = train_test_split(x, y_train, test_size = 0.3, random_state = 0)
Y_Test = pd.DataFrame(Y_test)

Y_Test.head()
Y_TEST = Y_Test.idxmax(axis=1)

Y_TEST.head()
random.seed(30)
    #initializing ANN

classifier = Sequential()



    # Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 13 ))



    # Adding the second hidden layer

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh',))



#classifier.add(Dropout(0.2))



    # Adding the third hidden layer

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh'))



#classifier.add(Dropout(0.2))



    # Adding the output layer

classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'softmax'))



    # Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    

classifier.fit(X_train, Y_train, batch_size = 50, nb_epoch = 25)
y_pred = classifier.predict(X_test, verbose=True)

y_pred =pd.DataFrame(y_pred)
y_Pred = y_pred.idxmax(axis=1)

y_Pred.head()
print("Accuracy in Test set=",accuracy_score(Y_TEST, y_Pred))
resultant_velocity1 = (xtest['angular_velocity_X']**2+ xtest['angular_velocity_Y']**2+ xtest['angular_velocity_Z']**2)**.5

resultant_velocity1= pd.DataFrame(resultant_velocity1, columns = ['resultant_velocity1'])

resultant_acc1 = (xtest['linear_acceleration_X']**2+

xtest['linear_acceleration_Y']**2+  

xtest['linear_acceleration_Z']**2)**.5

#resultant_acc= np.transpose(np.matrix(np.array(resultant_acc.T)))

resultant_acc1= pd.DataFrame(resultant_acc1, columns = ['resultant_acc1'])

power1=resultant_velocity1['resultant_velocity1']*resultant_acc1['resultant_acc1']

#As floor is fuction of friction, To estimate the friction factor differ force eq is required. Power is one of the factor on which friction force depends

power1= pd.DataFrame(power1, columns = ['power1'])

resultant_velocity1.head()
resultant_acc1.head()
power1.head()
#xt = xtest.iloc[:,3].values

#yt = xtest.iloc[:,4].values

#zt = xtest.iloc[:,5].values

#wt = xtest.iloc[:,6].values



#b0 = +2.0 * (np.multiply(wt,xt) + np.multiply(yt , zt))

#b1 = +1.0 - 2.0 * (np.multiply(xt , xt) + np.multiply(yt , yt))

#b1.shape

#Xt = np.arctan2(b0,b1)



#b2 = +2.0 * (np.multiply(wt , yt) - np.multiply(zt , xt))



#Yt = np.arcsin(b2)



#b3 = +2.0 * (np.multiply(wt , zt) + np.multiply(xt , yt))

#b4 = +1.0 - 2.0 * (np.multiply(yt , yt) + np.multiply(zt , zt))

#Zt = np.arctan2(b3, b4)
#Xt.shape, Yt.shape, Zt.shape
#xtest['euler_rotation_x'] = Xt

#xtest['euler_rotation_y'] = Yt

#xtest['euler_rotation_z'] = Zt  

#xtest['resultant_angle'] = (xtest['euler_rotation_x'] ** 2 + xtest['euler_rotation_y'] ** 2 + xtest['euler_rotation_z'] ** 2) ** .5
xtest_new= pd.concat([xtest,resultant_velocity1, resultant_acc1, power1], axis=1)

x3 = xtest_new

xtest_new.head()



x3.head()
xtest_new = xtest_new.iloc[:, 3:16]

xtest_new = pd.DataFrame(xtest_new)

xtest_new.head()
xtest_new.shape
#xtest_new = xtest_new.groupby('series_id')['orientation_X','orientation_Y','orientation_Z','orientation_W', 'linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z', 'angular_velocity_X','angular_velocity_Y','angular_velocity_Z','resultant_velocity1','resultant_acc1','power1',].mean()

#xtest_new.head()

#xtest_new.shape
sc = StandardScaler()

xtest_new = sc.fit_transform(xtest_new)

xtest_new= pd.DataFrame(xtest_new)

xtest_new.head()

Y_test_final = classifier.predict(xtest_new)

Y_test_final=pd.DataFrame(Y_test_final)

Y_test_final.head()
Y_test_final.shape
Y_test_final.columns = ['carpet','concrete','fine_concrete','hard_tiles', 'hard_tiles_large_space', 'soft_pvc', 'soft_tiles','tiled','wood']
Y_test_final.head()
Y_test_final.shape
Y_test_final = Y_test_final.idxmax(axis=1)

Y_test_final.head()
Y_test_final= pd.DataFrame(Y_test_final, columns = ['surface'])

Y_test_final.head()
seriesid_f = x3.iloc[:,1]

seriesid_f.head()
seriesid_f= pd.DataFrame(seriesid_f, columns = ['series_id'])

seriesid_f.head()
Y_test_final1 = pd.concat([seriesid_f, Y_test_final], axis=1)

Y_test_final1.head()
Y_test_final1.shape
#Y_test_final3 = Y_test_final1.pivot_table(values=["surface"], index=["series_id"], aggfunc=pd.mode)

Y_test_final3 = Y_test_final1.pivot_table(values=["surface"],

                                   index=["series_id"],

                                   aggfunc=lambda x: x.mode().iat[0])

Y_test_final3.shape
Y_test_final3.head()
#Y_test_final4= Y_test_final3.iloc[:,0]

#Y_test_final2 = Y_test_final1.groupby('series_id')['surface'].mode

#Y_test_final2.head()
#Y_test_final = Y_test_final.iloc[:, [1,17,18,19,20,21,22,23,24,25] ].values
#Y_test_final.head()
#Y_test_final = Y_test_final.groupby('series_id')['surface'].mode()

#Y_test_final.columns = ['carpet','concrete','fine_concrete','hard_tiles', 'hard_tiles_large_space', 'soft_pvc', 'soft_tiles','tiled','wood']
#Y_test_final = Y_test_final.idxmax(axis=1)

#Y_test_final.head()
#Y_test_final = pd.DataFrame(Y_test_final, columns = ['surface'])

#Y_test_final = Y_test_final.reset_index()

#Y_test_final.columns[0] = 'series_id'

#Y_test_final['series_id'] = Y_test_final.index

#Y_test_final.head()
Y_test_final3.to_csv("prediction.csv", index = True, index_label = 'series_id')