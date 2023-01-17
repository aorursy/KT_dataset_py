# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

dfmf1 = pd.read_csv("/kaggle/input/dataset/multi_outdoor_1_faulty.csv") #1
dfmf2 = pd.read_csv("/kaggle/input/dataset/multi_outdoor_2_faulty.csv") #2
dfmf3 = pd.read_csv("/kaggle/input/dataset/multi_indoor_3_faulty.csv") #3
dfmf4 = pd.read_csv("/kaggle/input/dataset/multi_indoor_4_faulty.csv") #4


#Drop NaN values
for dataFrame in (dfmf1,
                    dfmf2,
                    dfmf3,
                    dfmf4 ):
    dataFrame.columns=['Humidity', 'Temprature' , 'Label']

    
#mc1,mc3 mf1 mf3
#dfmc1 = dfmc1.apply (pd.to_numeric, errors='coerce')
#dfmc1 = dfmc1.dropna()

#dfmc3 = dfmc3.apply (pd.to_numeric, errors='coerce')
#dfmc3 = dfmc3.dropna()

dfmf1 = dfmf1.apply (pd.to_numeric, errors='coerce')
dfmf1 = dfmf1.dropna()

dfmf3 = dfmf3.apply (pd.to_numeric, errors='coerce')
dfmf3 = dfmf3.dropna()


combined1 =[ dfmf1.reset_index(drop=True),
             dfmf2.reset_index(drop=True),
             dfmf3.reset_index(drop=True),
             dfmf4.reset_index(drop=True)]
result_com1 = pd.concat(combined1)
print('Combined Single and multi Hop:\n', result_com1)

print(len(result_com1))

r = result_com1.reset_index()
result_com1= r.iloc[:,1:]
result_com1
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
## Normalisation

result_com1.rename(columns={ 0:'Humidity'
                            , 1:'Temprature'
                            , 2:'Label'}, 
                 inplace=True)

Normalised_S = result_com1.iloc[:, :-1]
Normalised_Single = preprocessing.normalize(Normalised_S)
Normalised_S_data = pd.DataFrame(Normalised_Single)
print('Normalised data for Single and multi hop:\n',Normalised_S_data)
print(Normalised_S_data.shape)
#Normalised_S_data
Normalised_S_labels = result_com1.iloc[:, 2]
Normalised_S_labels
len(Normalised_S_data)
from sklearn.model_selection import train_test_split

tr_data, ts_data, tr_label, ts_label = train_test_split(Normalised_S_data, Normalised_S_labels, test_size = 0.2, random_state = 42)
len(tr_data)
#type(tr_data)
len(ts_data)
len(tr_label)
len(ts_label)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
results = []
names = []


tr_data_arr = np.array(tr_data)

tr_label_arr = np.array(tr_label)

from sklearn.svm import SVC

clf = SVC(C = 20, kernel='rbf',random_state=42)
clf.fit(tr_data_arr, tr_label_arr)
prediction = clf.predict(np.array(ts_data))
print(prediction)
from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with SVM classifier:\n',Accuracy*100)



print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))

names.append('SVM')
results.append(Accuracy*100)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                             max_depth=30, max_features='auto', max_leaf_nodes=None,
                             min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=50, n_jobs=1,
                             oob_score=False, random_state=42,verbose=0, warm_start=False)

clf.fit(tr_data_arr, tr_label_arr)
#RandomForestClassifier(...)
prediction = (clf.predict(np.array(ts_data)))
print(prediction)
          
Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with RandomForestClassifier classifier:\n',Accuracy*100)
print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))
names.append('RandomForestClassifier')
results.append(Accuracy*100)



#Multiclass classification

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb

clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
clf.fit(tr_data_arr, tr_label_arr)
#RandomForestClassifier(...)
prediction = (clf.predict(np.array(ts_data)))
print(prediction)
          
Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with XGBoost classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))

names.append('XGBoost')
results.append(Accuracy*100)

from sklearn.ensemble import  GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators = 150 , random_state = 42 )
clf.fit(tr_data_arr, tr_label_arr)
prediction = (clf.predict(np.array(ts_data)))
print(prediction)
          
Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with GradientBoostingClassifier classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))

names.append('GradientBoostingClassifier')
results.append(Accuracy*100)

from sklearn.neighbors import KNeighborsClassifier 

clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf.fit(tr_data_arr, tr_label_arr)
prediction = (clf.predict(np.array(ts_data)))
print(prediction)
          
Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with KNN classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))

names.append('KNN')
results.append(Accuracy*100)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 48 , random_state = 42 )
clf.fit(tr_data_arr, tr_label_arr)
prediction = (clf.predict(np.array(ts_data)))
print(prediction)
          
Accuracy = accuracy_score(prediction, ts_label)
print('Accuracy with DecisionTreeClassifier classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(ts_label, prediction))
print('Classification Report')
print(classification_report(ts_label, prediction))

names.append('DecisionTreeClassifier')
results.append(Accuracy*100)



import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

name_df = pd.DataFrame(names) 
result_df = pd.DataFrame(results) 

name_df['Accuracy'] = result_df
name_df.rename(columns={ 0: 'Algorithm'
                   ,}, 
                 inplace=True)

print(name_df)
#print(result_df)


fig = px.line(name_df,  x='Algorithm', y='Accuracy')
fig.update_layout(yaxis_range = [0.00, 100.00],
                  title_text="Algorithm Comparison")
fig.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report

from sklearn.decomposition import PCA
from sklearn import tree
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras import Sequential
import random
import tensorflow
## Set Result for Reproducible Results
np.random.seed(1500)
tensorflow.random.set_seed(1500)
Tssdata = pd.read_csv("/kaggle/input/dataset/multi_outdoor_1_faulty.csv")
Trrdata= pd.read_csv("/kaggle/input/dataset/multi_outdoor_2_faulty.csv")

Data=np.concatenate((Tssdata,Trrdata),axis=0) #adding output to normalised data
Data=pd.DataFrame(Data)

Data.shape
Data
Tsdata = pd.read_csv("/kaggle/input/dataset/multi_indoor_3_faulty.csv")
Trdata= pd.read_csv("/kaggle/input/dataset/multi_indoor_4_faulty.csv")

Data1=np.concatenate((Tsdata,Trdata),axis=0) #adding output to normalised data
Data1=pd.DataFrame(Data1)
Data1
Finaldata=np.concatenate((Data,Data1),axis=0) #adding output to normalised data
Finaldata=pd.DataFrame(Finaldata)
#remove NaN 
Finaldata = Finaldata.apply (pd.to_numeric, errors='coerce')
Finaldata = Finaldata.dropna()
Finaldata
trainx = Finaldata.iloc[:,0:2].values
trainy = Finaldata.iloc[:,2].values
trainx

trainy
from sklearn.model_selection import train_test_split

tr_x, vl_x, y_train, y_test = train_test_split(trainx, trainy, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(tr_x)
X_test = sc.transform(vl_x)
y_train = y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train)
print(y_train)
X_train.shape
y_train.shape
y_test_real = y_test
y_test_real
y_test = y_test.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y_test = encoder.fit_transform(y_test)
print(y_test)
from keras.layers import Dense, Conv1D, Conv2D,LSTM,Flatten, Embedding, Dropout

accuracy = []
loss = []
name = []
epochs = []

model1 = Sequential()
model1.add(Dense(10, input_shape=(2,), activation='relu', name='fc1'))
model1.add(Dense(10, activation='relu', name='fc2'))
model1.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model1.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model1.summary())

m1 = model1.fit(X_train, y_train, verbose=2, epochs=10)#, epochs=200)

results1 = model1.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results1[0]))
print('Final test set accuracy: {:4f}'.format(results1[1]))


accuracy1 = m1.history["accuracy"]
loss1 = m1.history['loss']
name.append('RNN1')
#accuracy1.shape

acc_RNN1 = ('{:4f}'.format(results1[1]))
             
print(accuracy1)
print(loss1)
#print(acc_RNN1)
model2 = Sequential()
model2.add(Dense(10, input_shape=(2,), activation='relu', name='fc1'))
model2.add(Dense(10, activation='relu', name='fc2'))
model2.add(Dense(10, activation='sigmoid'))
model2.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model2.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model2.summary())

m2 = model2.fit(X_train, y_train, verbose=2, epochs=10)#, epochs=200)

results2 = model2.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results2[0]))
print('Final test set accuracy: {:4f}'.format(results2[1]))


accuracy2 = m2.history["accuracy"]
loss2 = m2.history['loss']
name.append('RNN2')
acc_RNN2 = ('{:4f}'.format(results2[1]))

print(accuracy2)
print(loss2)
model3 = Sequential()
model3.add(Dense(10, input_shape=(2,), activation='relu', name='fc1'))
model3.add(Dense(10, activation='relu', name='fc2'))
model3.add(Dense(10, activation='relu'))
model3.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model3.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model3.summary())

m3 = model3.fit(X_train, y_train, verbose=2, epochs=10)#, epochs=200)

results3 = model3.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results3[0]))
print('Final test set accuracy: {:4f}'.format(results3[1]))


accuracy3 = m3.history["accuracy"]
loss3 = m3.history['loss']
name.append('RNN3')
acc_RNN3 = ('{:4f}'.format(results3[1]))

print(accuracy3)
print(loss3)
model4 = Sequential()
model4.add(Dense(10, input_shape=(2,), activation='relu', name='fc1'))
model4.add(Dense(10, activation='relu', name='fc2'))
model4.add(Dense(10, activation='relu'))
model4.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model4.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model4.summary())

m4 = model4.fit(X_train, y_train, verbose=2, batch_size = 5, epochs=10)#, epochs=200)

results4 = model4.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results4[0]))
print('Final test set accuracy: {:4f}'.format(results4[1]))


accuracy4 = m4.history["accuracy"]
loss4 = m4.history['loss']
name.append('RNN4')

print(accuracy4)
print(loss4)
acc_RNN4 = ('{:4f}'.format(results4[1]))

model5 = Sequential()
model5.add(Dense(100, input_shape=(2,), activation='relu', name='fc1'))
model5.add(Dense(100, activation='relu', name='fc2'))
model5.add(Dense(100, activation='relu'))
model5.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model5.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model5.summary())

m5 = model5.fit(X_train, y_train, verbose=2, epochs=10, batch_size = 5)

results5 = model5.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results5[0]))
print('Final test set accuracy: {:4f}'.format(results5[1]))


pred = model5.predict(X_test)
pred
predicted_labels =[]
for i in pred:
    label = np.argmax(i)
    predicted_labels.append(label)
#predicted_labels
accuracy5 = m5.history["accuracy"]
loss5 = m5.history['loss']
name.append('RNN5')

print(accuracy5)
print(loss5)

acc_RNN5 = ('{:4f}'.format(results5[1]))
var_acc_RNN5 = (results5[1]) * 100
#print(var_acc_RNN5)   
y_test_real

#Confution Matrix and Classification Report
print('Confusion Matrix')
print(confusion_matrix(y_test_real, predicted_labels))
print('Classification Report')
print(classification_report(y_test_real, predicted_labels))


model6 = Sequential()
model6.add(Dense(100, input_shape=(2,), activation='relu', name='fc1'))
model6.add(Dropout(0.2))
model6.add(Dense(100, activation='relu', name='fc2'))
model6.add(Dropout(0.2))
model6.add(Dense(100, activation='relu'))
model6.add(Dense(5, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model6.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model6.summary())

m6 = model6.fit(X_train, y_train, verbose=2, epochs=10, batch_size = 5)

results6 = model6.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results6[0]))
print('Final test set accuracy: {:4f}'.format(results6[1]))


accuracy6 = m6.history["accuracy"]
loss6 = m6.history['loss']
name.append('RNN6')

print(accuracy6)
print(loss6)

acc_RNN6 = ('{:4f}'.format(results6[1]))

#testing accuracies
testing_acc =[]
testing_acc.append(acc_RNN1) 
testing_acc.append(acc_RNN2) 
testing_acc.append(acc_RNN3)
testing_acc.append(acc_RNN4)
testing_acc.append(acc_RNN5)
testing_acc.append(acc_RNN6) 

print(testing_acc)
print(name)

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc6 = []

for i in accuracy1:
    i = i*100
    acc1.append(i)
    

for i in accuracy2:
    i = i*100
    acc2.append(i)
    
for i in accuracy3:
    i = i*100
    acc3.append(i)

for i in accuracy4:
    i = i*100
    acc4.append(i)

for i in accuracy5:
    i = i*100
    acc5.append(i)

for i in accuracy6:
    i = i*100
    acc6.append(i)

print(accuracy1)      
print(acc1)
import matplotlib.pyplot as pyplot
#graph for accuracy
pyplot.figure(figsize= (10, 10))
pyplot.plot(acc1, color='blue', linestyle='-', marker='o', markerfacecolor='blue', markersize=5)
pyplot.plot(acc2,color='coral', linestyle='-', marker='x', markerfacecolor='coral', markersize=5)
pyplot.plot(acc3, color='yellow', linestyle='-', marker='o', markerfacecolor='yellow', markersize=5)
pyplot.plot(acc4,color='pink', linestyle='-', marker='x', markerfacecolor='pink', markersize=5)
pyplot.plot(acc5, color='aqua', linestyle='-', marker='o', markerfacecolor='blue', markersize=5)
pyplot.plot(acc6,color='black', linestyle='-', marker='x', markerfacecolor='black', markersize=5)

pyplot.title("Accuracies of Neural networks")
pyplot.legend(['RNN1','RNN2','RNN3','RNN4','RNN5','RNN6'])
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracies")
epochs = [1,2,3,4,5,6,7,8,9,10]
pyplot.xticks(epochs)
#pyplot.grid(color='black', linestyle='-', linewidth=0.2)
pyplot.show()
#graph for loss
pyplot.figure(figsize= (10, 10))
pyplot.plot(loss1, color='blue', linestyle='-', marker='o', markerfacecolor='blue', markersize=5)
pyplot.plot(loss2,color='coral', linestyle='-', marker='x', markerfacecolor='coral', markersize=5)
pyplot.plot(loss3, color='yellow', linestyle='-', marker='o', markerfacecolor='yellow', markersize=5)
pyplot.plot(loss4,color='pink', linestyle='-', marker='x', markerfacecolor='pink', markersize=5)
pyplot.plot(loss5, color='aqua', linestyle='-', marker='o', markerfacecolor='blue', markersize=5)
pyplot.plot(loss6,color='black', linestyle='-', marker='x', markerfacecolor='black', markersize=5)

pyplot.title("Losses in Neural networks")
pyplot.legend(['RNN1','RNN2','RNN3','RNN4','RNN5','RNN6'])
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
epochs = [1,2,3,4,5,6,7,8,9,10]
pyplot.xticks(epochs)
#pyplot.grid(color='black', linestyle='-', linewidth=0.2)
pyplot.show()
X_train.shape
y_train.shape
y_train
###Reshape the data
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
###Reshape the data
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
##Build the LSTM model
#model_lstm = Sequential()
#model_lstm.add(LSTM(100, return_sequences=True,input_shape = (2,1)))
#model_lstm.add(LSTM(100, return_sequences= False))
#model_lstm.add(Dense(10)) #can't use relu, sigmoid, not working
#model_lstm.add(Dense(5))
#optimizer = Adam(lr=0.001)
#model_lstm.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#print('Neural Network Model Summary: ')
#print(model_lstm.summary())
#
#lstm = model_lstm.fit(X_train, y_train, verbose=2, epochs=10, batch_size = 5)
#
#results_lstm = model_lstm.evaluate(X_test, y_test)
#
#print('Final test set loss: {:4f}'.format(results_lstm[0]))
#print('Final test set accuracy: {:4f}'.format(results_lstm[1]))

##Build the  model
#from keras.layers import Conv2D
#
#model_lstm = Sequential()
#model_lstm.add(Conv1D(32, kernel_size= 3, input_shape = (1,2)))
##model_lstm.add(LSTM(100, return_sequences=True,input_shape = (2,1)))
#model_lstm.add(Flatten())
#model_lstm.add(Dense(10)) #can't use relu, sigmoid, not working
#model_lstm.add(Dense(5))
#optimizer = Adam(lr=0.001)
#model_lstm.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#print('Neural Network Model Summary: ')
#print(model_lstm.summary())
#
#lstm = model_lstm.fit(X_train, y_train, verbose=2, epochs=10, batch_size = 5)
#
#results_lstm = model_lstm.evaluate(X_test, y_test)
#
#print('Final test set loss: {:4f}'.format(results_lstm[0]))
#print('Final test set accuracy: {:4f}'.format(results_lstm[1]))
l1_data = pd.read_csv("/kaggle/input/dataset-label/multi_outdoor_1_faulty_label_1.csv") 

#remove NaN 
l1_data = l1_data.apply (pd.to_numeric, errors='coerce')
l1_data = l1_data.dropna()
l1_data.shape
l1_trainx = l1_data.iloc[:,0:2].values
l1_trainy = l1_data.iloc[:,2].values
l1_tr_x, l1_vl_x, l1_y_train, l1_y_test = train_test_split(l1_trainx, l1_trainy, test_size = 0.2, random_state = 0)
#scale

sc = StandardScaler()
l1_X_train = sc.fit_transform(l1_tr_x)
l1_X_test = sc.transform(l1_vl_x)
l1_y_train = l1_y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l1_y_train = encoder.fit_transform(l1_y_train)
print(l1_y_train)

l1_X_train.shape
l1_y_train.shape
l1_y_test = l1_y_test.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l1_y_test = encoder.fit_transform(l1_y_test)
print(l1_y_test)
test_acc_labels = []
test_loss_labels = []
test_name_labels = []
l1_model5 = Sequential()
l1_model5.add(Dense(100, input_shape=(2,), activation='relu', name='fc1'))
l1_model5.add(Dense(100, activation='relu', name='fc2'))
l1_model5.add(Dense(100, activation='relu'))
l1_model5.add(Dense(2, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
l1_model5.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#print('Neural Network Model Summary: ')
#print(l1_model5.summary())


l1_m5 = l1_model5.fit(l1_X_train, l1_y_train, verbose=2, epochs=10, batch_size = 5)

l1_results5 = l1_model5.evaluate(l1_X_test, l1_y_test)

print('Final test set loss: {:4f}'.format(l1_results5[0]))
print('Final test set accuracy: {:4f}'.format(l1_results5[1]))

test_acc_labels.append((l1_results5[1])*100)
test_loss_labels.append(l1_results5[0])
test_name_labels.append('Random fault')

print('test_name_labels:',test_name_labels)
print('test_acc_labels:', test_acc_labels)
print('test_loss_labels:',test_loss_labels)

accuracy_l1 = l1_m5.history["accuracy"]
loss_l1 = l1_m5.history['loss']
l2_data = pd.read_csv("/kaggle/input/dataset-label/multi_outdoor_2_faulty_label_2.csv") 

#remove NaN 
l2_data = l2_data.apply (pd.to_numeric, errors='coerce')
l2_data = l2_data.dropna()
l2_data.shape

l2_trainx = l2_data.iloc[:,0:2].values
l2_trainy = l2_data.iloc[:,2].values


l2_tr_x, l2_vl_x, l2_y_train, l2_y_test = train_test_split(l2_trainx, l2_trainy, test_size = 0.2, random_state = 0)


#scale

sc = StandardScaler()
l2_X_train = sc.fit_transform(l2_tr_x)
l2_X_test = sc.transform(l2_vl_x)
l2_y_train = l2_y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l2_y_train = encoder.fit_transform(l2_y_train)
print(l2_y_train)

l2_X_train.shape
l2_y_train.shape

l2_y_test = l2_y_test.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l2_y_test = encoder.fit_transform(l2_y_test)
print(l2_y_test)


l2_m5 = l1_model5.fit(l2_X_train, l2_y_train, verbose=2, epochs=10, batch_size = 5)

l2_results5 = l1_model5.evaluate(l2_X_test, l2_y_test)

print('Final test set loss: {:4f}'.format(l2_results5[0]))
print('Final test set accuracy: {:4f}'.format(l2_results5[1]))

test_acc_labels.append((l2_results5[1])*100)
test_loss_labels.append(l2_results5[0])
test_name_labels.append('Malfunction fault')

print('test_name_labels:',test_name_labels)
print('test_acc_labels:', test_acc_labels)
print('test_loss_labels:',test_loss_labels)

accuracy_l2 = l2_m5.history["accuracy"]
loss_l2 = l2_m5.history['loss']
l3_data = pd.read_csv("/kaggle/input/dataset-label/multi_indoor_3_faulty_label_3.csv")

#remove NaN 
l3_data = l3_data.apply (pd.to_numeric, errors='coerce')
l3_data = l3_data.dropna()
l3_data.shape

l3_trainx = l3_data.iloc[:,0:2].values
l3_trainy = l3_data.iloc[:,2].values


l3_tr_x, l3_vl_x, l3_y_train, l3_y_test = train_test_split(l3_trainx, l3_trainy, test_size = 0.2, random_state = 0)


#scale

sc = StandardScaler()
l3_X_train = sc.fit_transform(l3_tr_x)
l3_X_test =      sc.transform(l3_vl_x)
l3_y_train = l3_y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l3_y_train = encoder.fit_transform(l3_y_train)
print(l3_y_train)

l3_X_train.shape
l3_y_train.shape

l3_y_test = l3_y_test.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l3_y_test = encoder.fit_transform(l3_y_test)
print(l3_y_test)


l3_m5 = l1_model5.fit(l3_X_train, l3_y_train, verbose=2, epochs=10, batch_size = 5)

l3_results5 = l1_model5.evaluate(l3_X_test, l3_y_test)

print('Final test set loss: {:4f}'.format(l3_results5[0]))
print('Final test set accuracy: {:4f}'.format(l3_results5[1]))

test_acc_labels.append((l3_results5[1])*100)
test_loss_labels.append(l3_results5[0])
test_name_labels.append('Drift fault')

print('test_name_labels:',test_name_labels)
print('test_acc_labels:', test_acc_labels)
print('test_loss_labels:',test_loss_labels)

accuracy_l3 = l3_m5.history["accuracy"]
loss_l3 = l3_m5.history['loss']
l4_data = pd.read_csv("/kaggle/input/dataset-label//multi_indoor_4_faulty_label_4.csv")

#remove NaN 
l4_data = l4_data.apply (pd.to_numeric, errors='coerce')
l4_data = l4_data.dropna()
l4_data.shape

l4_trainx = l4_data.iloc[:,0:2].values
l4_trainy = l4_data.iloc[:,2].values


l4_tr_x, l4_vl_x, l4_y_train, l4_y_test = train_test_split(l4_trainx, l4_trainy, test_size = 0.2, random_state = 0)


#scale

sc = StandardScaler()
l4_X_train = sc.fit_transform(l4_tr_x)
l4_X_test =      sc.transform(l4_vl_x)
l4_y_train = l4_y_train.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l4_y_train = encoder.fit_transform(l4_y_train)
print(l4_y_train)

l4_X_train.shape
l4_y_train.shape

l4_y_test = l4_y_test.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
l4_y_test = encoder.fit_transform(l4_y_test)
print(l4_y_test)


l4_m5 = l1_model5.fit(l4_X_train, l4_y_train, verbose=2, epochs=10, batch_size = 5)

l4_results5 = l1_model5.evaluate(l4_X_test, l4_y_test)

print('Final test set loss: {:4f}'.format(l4_results5[0]))
print('Final test set accuracy: {:4f}'.format(l4_results5[1]))

test_acc_labels.append((l4_results5[1])*100)
test_loss_labels.append(l4_results5[0])
test_name_labels.append('Bias fault')

print('test_name_labels:',test_name_labels)
print('test_acc_labels:', test_acc_labels)
print('test_loss_labels:',test_loss_labels)

accuracy_l4 = l4_m5.history["accuracy"]
loss_l4 = l4_m5.history['loss']
label_acc1 = []
label_acc2 = []
label_acc3 = []
label_acc4 = []

for i in accuracy_l1:
    i = i*100
    label_acc1.append(i)
    

for i in accuracy_l2:
    i = i*100
    label_acc2.append(i)
    
for i in accuracy_l3:
    i = i*100
    label_acc3.append(i)

for i in accuracy_l4:
    i = i*100
    label_acc4.append(i)


#graph for accuracy as per separate faults
pyplot.figure(figsize= (10, 10))
pyplot.plot(label_acc1, color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
pyplot.plot(label_acc2,color='black', linestyle='dashed', marker='x', markerfacecolor='black', markersize=10)
pyplot.plot(label_acc3, color='red', linestyle= 'dashed', marker='o', markerfacecolor='red', markersize=10)
pyplot.plot(label_acc4,color='green', linestyle='dashed', marker='x', markerfacecolor='green', markersize=10)

pyplot.title("Accuracies of Neural networks")
pyplot.legend(['Random fault','Malfunction fault','Drift fault','Bias fault'])
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracies")
epochs = [1,2,3,4,5,6,7,8,9,10]
pyplot.xticks(epochs)
pyplot.grid(color='black', linestyle='-', linewidth=0.2)
pyplot.show()
var_acc_RNN5
name_RNN5 = []
RNN_acc = []
RNN_acc.append(var_acc_RNN5)
name_RNN5.append('RNN')
df1 = pd.DataFrame(name_RNN5)
df2 = pd.DataFrame(RNN_acc)


df1['Accuracy'] = df2
df1.rename(columns={ 0: 'Algorithm'
                   ,}, 
                 inplace=True)

print(df1)

result_df
var_df = name_df

name_df1 = name_df.append(df1)
name_df2 = name_df1.reset_index(drop=True)
name_df2 
# Draw plot

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=name_df2.index, ymin=0, ymax= name_df2.Accuracy, color='red', alpha=0.7, linewidth=2)
ax.scatter(x=name_df2.index, y=name_df2.Accuracy, s=200, color='green', alpha=0.7)


# Title, Label, Ticks and Ylim
ax.set_title('Plot for Algorithms comparision', fontdict={'size':22})
ax.set_ylabel('Testing Accuracy')
ax.set_xticks(name_df2.index)
ax.set_xticklabels(name_df2.Algorithm, rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(0, 100)

# Annotate
for row in name_df2.itertuples():
    ax.text(row.Index, row.Accuracy+.5, s=round(row.Accuracy, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

plt.show()
