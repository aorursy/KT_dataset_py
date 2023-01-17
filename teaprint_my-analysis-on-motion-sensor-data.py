# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/A_DeviceMotion_data/A_DeviceMotion_data"))

# Any results you write to the current directory are saved as output.
# Data Folders:
Folders = glob('../input/A_DeviceMotion_data/A_DeviceMotion_data/*_*')
Folders = [s for s in Folders if "csv" not in s]

Df_all_list = []
Exp = 0
# Segment the data to 400 sampels frames , each one will be a different Expirament
Segment_Size = 400

# Activety types dict:
activity_codes = {'dws':1,'jog':2,'sit':3,'std':4,'ups':5,'wlk':6}        
activity_types = list(activity_codes.keys())

# Load All data:
for j  in Folders:
    Csv = glob(j + '/*' )


    for i in Csv:
        df = pd.read_csv(i)
        # Add Activety label, Subject name and Experiment number
        df['Activity'] = activity_codes[j[49:52]]
        df['Sub_Num'] = i[len(j)+5:-4]
        df['Exp_num'] = 1
        ExpNum = np.zeros((df.shape[0])) 
        for i in range(0,df.shape[0]-Segment_Size,Segment_Size):
            ExpNum[range(i,i+Segment_Size)] = i/Segment_Size +Exp*100 
        df['Exp_num'] = ExpNum
        #Df_all = pd.concat([Df_all,df])
        Df_all_list.append(df)
        Exp += 1        

Df_all = pd.concat(Df_all_list,axis=0)  
print(Df_all.shape)
print(Df_all.columns)
np.unique(Df_all['Sub_Num'])
### Missing values
checks = pd.isna(Df_all).sum()
print(checks)
### class balance
class_counts = list()

for act in activity_types[:1]:
    class_counts.append(Df_all[Df_all['Activity']==activity_codes[act]].count())
plt.figure(1)
plt.title('Size of each class')
plt.xlabel('activity type')
plt.hist(Df_all['Activity'],bins=range(1,8),rwidth=0.5,align='left')

### Length of time series
series_length = list()
for act in activity_types:
    for sub in range(1,25):
        sub = str(sub)
        series_length.append(Df_all[(Df_all['Sub_Num']==sub) & (Df_all['Activity']==activity_codes[act])].shape[0])
plt.figure(2)
plt.title('Histogram of length of raw time series')
plt.hist(series_length,rwidth=0.5,align='left')


### show the first few records of motion type
plt.figure(3)
colors = ['r','g','b','c','m','y','k']
for act in activity_types:
    plt.subplot('61'+str(activity_codes[act]))
    plt.subplots_adjust(hspace=1.0)
    df = Df_all[(Df_all['Sub_Num']=='1') & (Df_all['Activity']==activity_codes[act])]
    plt.title(act)
    plt.plot(df['userAcceleration.z'][:400])
    plt.xticks([]) # turn off x labels
    plt.yticks([])  # turn off y labels



#  Calculate features
df_sum = Df_all.groupby('Exp_num', axis=0).mean().reset_index()
df_sum.columns = df_sum.columns.str.replace('.','_sum_')

df_sum_SS = np.power(Df_all.astype(float),2).groupby('Exp_num', axis=0).median().reset_index() 
df_sum_SS.columns = df_sum_SS.columns.str.replace('.','_sumSS_')

df_max = Df_all.groupby('Exp_num', axis=0).max().reset_index()
df_max.columns = df_max.columns.str.replace('.','_max_')

df_min = Df_all.groupby('Exp_num', axis=0).min().reset_index()
df_min.columns = df_min.columns.str.replace('.','_min_')

df_skew = Df_all.groupby('Exp_num', axis=0).skew().reset_index()
df_skew.columns = df_skew.columns.str.replace('.','_skew_')

df_std = Df_all.groupby('Exp_num', axis=0).std().reset_index()
df_std.columns = df_std.columns.str.replace('.','_std_')

Df_Features = pd.concat([ df_max , df_sum[df_sum.columns[2:-2]], 
                         df_min[df_min.columns[2:-2]], df_sum_SS[df_sum_SS.columns[2:-2]], 
                         df_std[df_std.columns[2:-2]], df_skew[df_skew.columns[2:-2]]], axis=1)

X = Df_Features.drop(['Exp_num','Unnamed: 0','Activity','Sub_Num'],axis=1)
Y = Df_Features['Activity']

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#### dimension reduction
### use pca to reduce the dimension to 2D directly.
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

plt.figure(1)
colors = ['r','g','b','c','m','y','k']
lw = 2

for color, i, target_name in zip(colors, range(6), activity_types):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Use PCA directly')

### sklearn tsne
# sites.google.com/s/1HAV-HEiBhPHLgdh5Ejmu31TrVIQqw9HU/p/1bPpOCDlxW7i5nOpy3bvpnmkqa8Y-SDVa/edit
# Scale data
scl = StandardScaler()
scaled_X = scl.fit_transform(X)

# Reduce dimensions before feeding into tsne
pca = PCA(n_components=0.9, random_state=3)
pca_transformed = pca.fit_transform(scaled_X)

# Transform data
tsne = TSNE(random_state=3)
tsne_transformed = tsne.fit_transform(pca_transformed)

plt.figure(2)
for color, i, target_name in zip(colors, range(6), activity_types):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('TSNE')

series_list = list()
labels_list = list()
for act in activity_types:

    df = Df_all[(Df_all['Sub_Num']=='1') & (Df_all['Activity']==activity_codes[act])]
    series_list.append(df['userAcceleration.z'][:400])
    labels_list.extend([act])

#print(labels_list)
from dtaidistance import dtw
import numpy as np
from dtaidistance import clustering

series = np.array(series_list)
ds = dtw.distance_matrix_fast(series)

model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
model.fit(series)

model.plot(show_ts_label=labels_list,
           show_tr_label=True)

import itertools

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
import keras

#### Construct neural Architeture for baseline model
input_dim = X.shape[1]
input_img = Input(shape=(input_dim,))
d = Dense(50, activation='relu')(input_img)
d = Dense(20, activation='relu')(d)
output = Dense(7, activation='softmax', kernel_regularizer=l2(0.01))(d)
model = Model(input_img,output)
model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
# One-hot encoding
Y = to_categorical(Y)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

history = model.fit(X_train, Y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=[X_test,Y_test])
#### Check model performance
### Overall test accuracy
score = model.evaluate(X_test, Y_test)
print ('keras test accuracy score:', score[1])


# One-hot decoding
y_pred = np.argmax(model.predict(X_test),axis=1)
y_test = np.argmax(Y_test,axis=1)
#print(y_pred,y_test)

correct = np.nonzero(y_pred==y_test)[0]
incorrect = np.nonzero(y_pred!=y_test)[0]
#print(correct)

### Check the correctly-predicted samples
plt.figure(1)
for i, cor in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.plot(X_test.iloc[cor,:])
    plt.title("Predicted:{}\nTrue:{}".format(activity_types[y_pred[cor]-1], 
                                              activity_types[y_test[cor]-1]))
    plt.xticks([]) # turn off x labels
    plt.yticks([])  # turn off y labels
    #plt.tight_layout()
plt.show()
### Check the incorrectly-predicted samples
plt.figure(2)
for i, cor in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.subplots_adjust(hspace=1.0,wspace=1.0)
    plt.plot(X_test.iloc[cor,:])
    plt.title("Predicted:{}\nTrue:{}".format(activity_types[y_pred[cor]-1], 
                                              activity_types[y_test[cor]-1]))
    plt.xticks([]) # turn off x labels
    plt.yticks([])  # turn off y labels
    #plt.tight_layout()
plt.show()
### Confusion matrix (predictive performance on different classes)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    function provided by sklearn example
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=activity_types,
                      title='Confusion matrix, without normalization')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

clf1 = RandomForestClassifier(n_estimators=100, max_depth=None,
     min_samples_split=2, random_state=0)
clf1.fit(X_train, Y_train)
featureImportance = clf1.feature_importances_

# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
feature_names = X.columns
idxSorted = np.argsort(featureImportance)[-10:]
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, feature_names[idxSorted])
plt.xlabel('Variable Importance')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.show()
# Need to one-hot decode before feding into GBM
y_train = np.argmax(Y_train,axis=1)
clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0)
clf2.fit(X_train, y_train)
featureImportance = clf2.feature_importances_


# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
feature_names = X.columns
idxSorted = np.argsort(featureImportance)[-10:]
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, feature_names[idxSorted])
plt.xlabel('Variable Importance')
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plt.show()

### One-hot encoding for tree classifier
print(Y_train.shape)
# Let target label become subjects
Y1 = Df_Features['Sub_Num'].iloc[:,0]
#### Construct neural Architeture for baseline model
input_dim = X.shape[1]
input_img = Input(shape=(input_dim,))
d = Dense(50, activation='relu')(input_img)
d = Dense(20, activation='relu')(d)
output = Dense(25, activation='softmax', kernel_regularizer=l2(0.01))(d)
model = Model(input_img,output)
model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
# One-hot encoding

y1 = to_categorical(Y1)
print(Y1.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, y1, test_size=0.3, random_state=0)

history = model.fit(X_train, Y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=[X_test,Y_test])

### Overall test accuracy
score = model.evaluate(X_test, Y_test)
print ('keras test accuracy score:', score[1])

for i in range(1,7):
    df = Df_Features[Df_Features['Activity']==i]
    x = df.drop(['Exp_num','Unnamed: 0','Activity','Sub_Num'],axis=1)
    y = df['Sub_Num'].iloc[:,0]
    y = to_categorical(y)
    
    #### Construct neural Architeture
    input_dim = x.shape[1]
    input_img = Input(shape=(input_dim,))
    d = Dense(50, activation='relu')(input_img)
    d = Dense(20, activation='relu')(d)
    output = Dense(25, activation='softmax', kernel_regularizer=l2(0.01))(d)
    model = Model(input_img,output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['categorical_accuracy'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    history = model.fit(X_train, Y_train,
                    epochs=1000,
                    batch_size=100,
                    shuffle=True,
                    verbose=0,
                    validation_data=[X_test,Y_test])

    ### Overall test accuracy
    score = model.evaluate(X_test, Y_test)
    print('Activity type:', activity_types[i-1])
    print('keras test accuracy score:', score[1])
mapping = {
    '1': 1,
    '2': 1,
    '3': 0,
    '4': 1,
    '5': 0,
    '6':1,
    '7':0,
    '8':0,
    '9':1,
    '10':0,
    '11':1,
    '12':1,
    '13':1,
    '14':1,
    '15':1,
    '16':0,
    '17':1,
    '18':0,
    '19':0,
    '20':1,
    '21':1,
    '22':1,
    '23':0,
    '24':0

}
# Create gender labels
Y2 = [mapping[i] for i in Df_Features['Sub_Num'].iloc[:,0]]
y2 = to_categorical(Y2)

X_train, X_test, Y_train, Y_test = train_test_split(X, y2, test_size=0.3, random_state=0)
print('The number of training samples:',X_train.shape[0])
print('The number of test samples:',X_test.shape[0])


#### Construct neural Architeture for baseline model
input_dim = x.shape[1]
input_img = Input(shape=(input_dim,))
d = Dense(50, activation='relu')(input_img)
d = Dense(20, activation='relu')(d)
output = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(d)
model = Model(input_img,output)
model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
    
history = model.fit(X_train, Y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=[X_test,Y_test])

### Overall test accuracy
score = model.evaluate(X_test, Y_test)
print ('keras test accuracy score:', score[1])
mapping = {
    '1': 1,
    '2': 0,
    '3': 0,
    '4': 1,
    '5': 0,
    '6':1,
    '7':0,
    '8':0,
    '9':1,
    '10':0,
    '11':0,
    '12':0,
    '13':0,
    '14':0,
    '15':0,
    '16':1,
    '17':1,
    '18':0,
    '19':1,
    '20':1,
    '21':0,
    '22':1,
    '23':0,
    '24':1

}
# Create gender labels
Y3 = [mapping[i] for i in Df_Features['Sub_Num'].iloc[:,0]]
y3 = to_categorical(Y3)

X_train, X_test, Y_train, Y_test = train_test_split(X, y3, test_size=0.3, random_state=0)
print('The number of training samples:',X_train.shape[0])
print('The number of test samples:',X_test.shape[0])


#### Construct neural Architeture for baseline model
input_dim = x.shape[1]
input_img = Input(shape=(input_dim,))
d = Dense(50, activation='relu')(input_img)
d = Dense(20, activation='relu')(d)
output = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(d)
model = Model(input_img,output)
model.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['categorical_accuracy'])
    
history = model.fit(X_train, Y_train,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=[X_test,Y_test])

### Overall test accuracy
score = model.evaluate(X_test, Y_test)
print ('keras test accuracy score:', score[1])