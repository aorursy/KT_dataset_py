# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools

import plotly as py

import plotly.graph_objs as go

init_notebook_mode(connected=True)







from sklearn import metrics

from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler,  OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.metrics import classification_report



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, MaxPooling1D , GlobalMaxPool1D , GlobalMaxPooling1D , GlobalAveragePooling1D , MaxPooling1D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical 

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import math

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
crime = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv' , encoding='latin-1')

code = pd.read_csv('/kaggle/input/crimes-in-boston/offense_codes.csv' , encoding='latin-1')

data = pd.read_csv('/kaggle/input/crimes-in-boston/crime.csv' , encoding='latin-1')

crime.info()

code.info()
crime.head()
code.head()
crime = crime.drop(['INCIDENT_NUMBER' ,  'REPORTING_AREA' , 'OFFENSE_DESCRIPTION' , 'OCCURRED_ON_DATE' , 'UCR_PART' ,'Location'] , axis=1)

crime.head()
crime.isnull().sum()
code.isnull().sum()
crime.SHOOTING = [1 if each == 'Y' else 0 for each in crime.SHOOTING]
crime = crime.dropna(how="any")

crime.isnull().sum()
crime.head(19)
le = LabelEncoder()

crime['DAY_OF_WEEK'] = le.fit_transform(crime['DAY_OF_WEEK'] ) 

crime['STREET'] = le.fit_transform(crime['STREET'])

crime['OFFENSE_CODE_GROUP'] = le.fit_transform(crime['OFFENSE_CODE_GROUP'])

crime['DISTRICT'] = le.fit_transform(crime['DISTRICT'])

crime.head()
def off_code_size(df):

    codes = pd.unique(df.OFFENSE_CODE)

    size = []

    for i in codes:

        size.append(len(df[df['OFFENSE_CODE'] == i]))

    return size



def code_name(df):

    codes = code.values

    names = []

    off_codes = pd.unique(df.OFFENSE_CODE)

    iterr = 0

    for i in off_codes:

        if(i in codes):

            iterr += 1

            names.append(codes[iterr][1])

    return names
from wordcloud import WordCloud

plt.figure(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(data.OFFENSE_CODE_GROUP))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(crime.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.savefig("corr.png")

plt.show()
crime_2015 = crime[crime.YEAR == 2015] 

crime_2016 = crime[crime.YEAR == 2016]

crime_2017 = crime[crime.YEAR == 2017]

crime_2018 = crime[crime.YEAR == 2018]
fig, ax1 = plt.subplots(2, 2, figsize= (140, 70) )



ax1[0,0].bar(code_name(crime_2015) , off_code_size(crime_2015))

ax1[0,0].tick_params(labelrotation=90)

ax1[0,1].bar(code_name(crime_2016) , off_code_size(crime_2016))  

ax1[0,1].tick_params(labelrotation=90)

ax1[1,0].bar(code_name(crime_2017) , off_code_size(crime_2017)) 

ax1[1,0].tick_params(labelrotation=90)

ax1[1,1].bar(code_name(crime_2018) , off_code_size(crime_2018)) 

ax1[1,1].tick_params(labelrotation=90)



ax1[0,0].grid()

ax1[0,1].grid()

ax1[1,0].grid()

ax1[1,1].grid()



plt.show()
sns.countplot(data=crime, x='DISTRICT')
sns.countplot(data=crime, x='MONTH')
sns.countplot(data=crime, x='YEAR')
sns.countplot(data=crime, x='HOUR')
trace1 = go.Scatter3d(

    x=crime_2018.OFFENSE_CODE,

    y=off_code_size(crime_2018),

    z=crime_2018.DISTRICT,

    name='2018',

    mode='markers',

    marker=dict(

        size=12,

        line=dict(

            color='rgb(120,120,120)',

            width=0.5

        ),

        opacity=0.7

    )

)



trace2 = go.Scatter3d(

    x=crime_2017.OFFENSE_CODE,

    y=off_code_size(crime_2017),

    z=crime_2017.DISTRICT,

    name='2017',

    mode='markers',

    marker=dict(

        size=12,

        symbol='circle',

        line=dict(

            color='rgb(160,160,160)',

            width=0.5

        ),

        opacity=0.7

    )

)

trace3 = go.Scatter3d(

    x=crime_2016.OFFENSE_CODE,

    y=off_code_size(crime_2016),

    z=crime_2016.DISTRICT,

    name='2016',

    mode='markers',

    marker=dict(

        size=12,

        line=dict(

            color='rgb(200,220,220)',

            width=0.5

        ),

        opacity=0.7

    )

)



trace4 = go.Scatter3d(

    x=crime_2015.OFFENSE_CODE,

    y=off_code_size(crime_2015),

    z=crime_2015.DISTRICT,

    name='2015',

    mode='markers',

    marker=dict(

        color='rgb(127, 127, 127)',

        size=12,

        symbol='circle',

        line=dict(

            width=0.5

        ),

        opacity=0.7

       

    )

)

data = [trace1, trace2 , trace3 , trace4]

layout = go.Layout(

    

    margin=dict(

        l=100,

        r=100,

        b=100,

        t=100

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple-3d-scatter')
labels_dis = le.inverse_transform(crime['DISTRICT'])

ld = sorted(pd.unique(labels_dis))

print(ld)
X = crime.drop(['DISTRICT'] , axis=1)

Y = crime.DISTRICT



x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.33, random_state=0)



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

knn = KNeighborsClassifier(n_neighbors=77 , metric='minkowski')

knn.fit(X_train , y_train)

y_pred_knn = knn.predict(X_test)



cm_knn = confusion_matrix(y_test , y_pred_knn)

print('KNN')

print(cm_knn)
f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm_knn, xticklabels=ld, yticklabels=ld, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("knn.png")

plt.show()
label_error = 1 - np.diag(cm_knn) / np.sum(cm_knn, axis=1)

plt.figure(figsize=(25,10))

plt.bar(ld,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("knn_bar.png")

plt.show
print(classification_report(y_test, y_pred_knn))
dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train , y_train)

y_pred_dtc = dtc.predict(X_test)



cm_dtc = confusion_matrix(y_test,y_pred_dtc)

print('DTC')

print(cm_dtc)
f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm_dtc, xticklabels=ld, yticklabels=ld, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("dtc.png")

plt.show()
label_error = 1 - np.diag(cm_dtc) / np.sum(cm_dtc, axis=1)

plt.figure(figsize=(25,10))

plt.bar(ld,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("dtc_bar.png")

plt.show()
print(classification_report(y_test, y_pred_dtc))
rfc = RandomForestClassifier(criterion='entropy' , n_estimators=33)

rfc.fit(X_train , y_train)

y_pred_rfc = rfc.predict(X_test)



cm_rfc = confusion_matrix(y_test,y_pred_rfc)

print('RFC')

print(cm_rfc)
f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm_rfc, xticklabels=ld, yticklabels=ld, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("rfc.png")

plt.show()
label_error = 1 - np.diag(cm_rfc) / np.sum(cm_rfc, axis=1)

plt.figure(figsize=(25,10))

plt.bar(ld,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("rfc_bar.png")

plt.show()
print(classification_report(y_test, y_pred_rfc))
gnb = GaussianNB()

gnb.fit(X_train , y_train)

y_pred_gnb = gnb.predict(X_test)



cm_gnb = confusion_matrix(y_test,y_pred_gnb)

print('GNB')

print(cm_gnb)
f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm_gnb, xticklabels=ld, yticklabels=ld, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("gbn.png")

plt.show()
label_error = 1 - np.diag(cm_gnb) / np.sum(cm_gnb, axis=1)

plt.figure(figsize=(25,10))

plt.bar(ld,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("gbn_bar.png")

plt.show()
print(classification_report(y_test, y_pred_gnb))
X = crime.drop(['DISTRICT'] , axis=1)

Y = crime['DISTRICT']



x_orjinal_train , x_orjinal_test, y_orjinal_train, y_orjinal_test = train_test_split(X, Y, test_size=0.33,random_state=21)



y_train = to_categorical(y_orjinal_train, num_classes = 12)

y_test = to_categorical(y_orjinal_test, num_classes = 12)



min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x_orjinal_train)

x_train = pd.DataFrame(x_scaled)

x_scaled1 = min_max_scaler.fit_transform(x_orjinal_test)

x_test = pd.DataFrame(x_scaled1)

x_train = x_train.values

x_test = x_test.values
x_train = np.asarray(x_train)

x_test = np.asarray(x_test)



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std
x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.33, random_state = 21)

x_train.shape
type(x_train)

print(x_train.shape)
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Activation, Dense

def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'swish': Activation(swish )})

classifier = Sequential()

classifier.add(Conv1D(110, 10, activation= 'swish', padding= 'Same', input_shape = (1, 10)))

classifier.add(Conv1D(110, 10, activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling1D(1))

classifier.add(Dropout(0.25))

classifier.add(Conv1D(130, 10, activation= 'swish', padding= 'Same'))

classifier.add(Conv1D(130, 10, activation= 'swish', padding= 'Same'))

classifier.add(MaxPooling1D(1))

classifier.add(Dropout(0.25))

classifier.add(Conv1D(150, 10, activation= 'swish', padding= 'Same'))

classifier.add(Conv1D(150, 10, activation= 'swish', padding= 'Same'))

classifier.add(GlobalAveragePooling1D())

classifier.add(Dropout(0.35))



classifier.add(Dense(12, activation= 'softmax'))

classifier.summary()



classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])



epochs = 145

batch_size = 2000

history = classifier.fit(x_train , y_train , verbose=1 , batch_size=batch_size , epochs=epochs ,validation_data=(x_test, y_test) )
from keras.utils import plot_model

plot_model(classifier)
loss, accuracy = classifier.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = classifier.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc.png")

plt.show()



fig, ax1 = plt.subplots(figsize= (15, 10) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss.png")

plt.show()



Y_pred = classifier.predict(x_validate)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_validate,axis = 1) 

cm = confusion_matrix(Y_true, Y_pred_classes)

print(cm)
f,ax = plt.subplots(figsize=(18, 18))

ax = sns.heatmap(cm, xticklabels=ld, yticklabels=ld, linewidths=.5 , cbar=False , fmt="d" , annot=True)

plt.savefig("corr_cnn.png")

plt.show()
label_error = 1 - np.diag(cm) / np.sum(cm, axis=1)

plt.figure(figsize=(25,10))

plt.bar(ld,label_error)

plt.xlabel('True Label')

plt.ylabel('Classified incorrectly')

plt.savefig("cnn_bar.png")