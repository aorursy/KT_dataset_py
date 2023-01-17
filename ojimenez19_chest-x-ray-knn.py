# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn import metrics

from sklearn import preprocessing

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Variables preparation



image_size= 32



dir_base='/kaggle/input/chest-xray-pneumonia/chest_xray'

test_dir = '{}/{}'.format(dir_base, 'test')

train_dir = '{}/{}'.format(dir_base, 'train')

val_dir = '{}/{}'.format(dir_base, 'val')



test_dir_normal_cases = '{}/{}/{}'.format(dir_base, 'test','NORMAL')

train_dir_normal_cases = '{}/{}/{}'.format(dir_base, 'train','NORMAL')

val_dir_normal_cases = '{}/{}/{}'.format(dir_base, 'val','NORMAL')



test_dir_pneumonia_cases = '{}/{}/{}'.format(dir_base, 'test','PNEUMONIA')

train_dir_pneumonia_cases = '{}/{}/{}'.format(dir_base, 'train','PNEUMONIA')

val_dir_pneumonia_cases = '{}/{}/{}'.format(dir_base, 'val','PNEUMONIA')
def get_data(path,label,data_type,process_flipped):

    result = []

    items = os.listdir(path)

    for i in items:

        img_path = '{}/{}/{}/{}'.format(dir_base,data_type,label,i)

        result.append((get_image_arr(img_path),label))

        if process_flipped == True:

            result.append((get_flipped_image_arr(img_path),label))

    return pd.DataFrame(result, columns=['image', 'label'],index=None)
def get_flipped_image_arr(path):

        original = cv2.imread(path) 

        img = cv2.flip(original, 1)

        img = cv2.resize(img, (image_size, image_size),interpolation = cv2.INTER_AREA).reshape(1,-1)

        return img
def get_image_arr(path):

        img = cv2.imread(path) 

        img = cv2.resize(img, (image_size, image_size),interpolation = cv2.INTER_AREA).reshape(1,-1)

        return img
def show_plot(k_values,accuracy_values):

    fig = plt.figure()

    fig.subplots_adjust(top=0.8)

    ax1 = fig.add_subplot()

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('K value')

    plt.plot(k_values,accuracy_values,label='Accuracy for k params')

    plt.scatter(k_values,accuracy_values,c=k_values,alpha=1)

    plt.legend()

    plt.show()
def show_confusion_matrix(model,x,y):

    disp = plot_confusion_matrix(model, x, y,

                             cmap=plt.cm.Blues,

                             normalize=None)

    plt.show()
data_normal = get_data(train_dir_normal_cases,'NORMAL','train',True)

data_pneumonia = get_data(train_dir_pneumonia_cases,'PNEUMONIA','train',False)
data = pd.concat([data_normal,data_pneumonia])

images = data.image

y = data.label



X = []

for i in images.values:

    X.append(i[0])

X = np.array(X)
print('X shape:',X.shape)

print('Y shape:',y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=30)
print('X train shape:',X_train.shape)

print('Y train shape:',y_train.shape)



print('X train shape:',X_test.shape)

print('Y train shape:',y_test.shape)
print('Classes: ',np.unique(y_train))
k_values = [1, 3, 5, 10, 20, 50, 100]

accuracy_values = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, predictions)

    accuracy_values.append(acc)

    print('Accuracy for k={}:'.format(str(k)),acc)

    print('\n')

    print(classification_report(y_test, predictions))

    print('=========================================================')

    print('\n')

show_plot(k_values,accuracy_values)
knn_model = KNeighborsClassifier(n_neighbors=10)

knn_model.fit(X,y)
#Load test data

test_data_normal = get_data(test_dir_normal_cases,'NORMAL','test',False)

test_data_pneumonia = get_data(test_dir_pneumonia_cases,'PNEUMONIA','test',False)



test_data_total = pd.concat([test_data_normal,test_data_pneumonia])



y_test_data = test_data_total.label



X_test_data = []

for i in test_data_total.image.values:

    X_test_data.append(i[0])

    

X_test_data = np.array(X_test_data)
predictions_test = knn_model.predict(X_test_data)

acc_test = metrics.accuracy_score(y_test_data, predictions_test)

print('Accuracy for test',acc_test)

print(classification_report(y_test_data, predictions_test))

show_confusion_matrix(knn_model,X_test_data,y_test_data)