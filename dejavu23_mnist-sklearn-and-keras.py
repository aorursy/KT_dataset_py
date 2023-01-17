import numpy as np 

import pandas as pd 



import random as rn



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



# plotly library

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, KFold 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import Pipeline

from scipy.stats import uniform



import itertools



import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



from keras.utils.np_utils import to_categorical

from keras.utils import np_utils



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

#from keras.layers import AvgPool2D, BatchNormalization, Reshape

from keras.optimizers import Adadelta, RMSprop, Adam

from keras.losses import categorical_crossentropy

from keras.wrappers.scikit_learn import KerasClassifier



import tensorflow as tf



import os

print(os.listdir("../input"))

img_rows, img_cols = 28, 28



np.random.seed(5)

#rn.seed(5)

#tf.set_random_seed(5)
def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_
def print_validation_report(y_true, y_pred):

    print("Classification Report")

    print(classification_report(y_true, y_pred))

    acc_sc = accuracy_score(y_true, y_pred)

    print("Accuracy : "+ str(acc_sc))

    

    return acc_sc
def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)

    #  square=True,

    plt.ylabel('true label')

    plt.xlabel('predicted label')
def plot_history_loss_and_acc(history_keras_nn):



    fig, axs = plt.subplots(1,2, figsize=(12,4))



    axs[0].plot(history_keras_nn.history['loss'])

    axs[0].plot(history_keras_nn.history['val_loss'])

    axs[0].set_title('model loss')

    axs[0].set_ylabel('loss')

    axs[0].set_xlabel('epoch')

    axs[0].legend(['train', 'validation'], loc='upper left')



    axs[1].plot(history_keras_nn.history['acc'])

    axs[1].plot(history_keras_nn.history['val_acc'])

    axs[1].set_title('model accuracy')

    axs[1].set_ylabel('accuracy')

    axs[1].set_xlabel('epoch')

    axs[1].legend(['train', 'validation'], loc='upper left')



    plt.show()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
y = train["label"]

X = train.drop(["label"],axis = 1)

X_test = test
X = X/255.0

X_test = X_test/255.0
# for best performance, especially of the NN classfiers,

# set mode = "commit"

mode = "edit"

mode = "commit"

#



if mode == "edit" :

    nr_samples = 1200



if mode == "commit" :    

    nr_samples = 30000



y_train=y[:nr_samples]

X_train=X[:nr_samples]

start_ix_val = nr_samples 

end_ix_val = nr_samples + int(nr_samples/3)

y_val=y[start_ix_val:end_ix_val]

X_val=X[start_ix_val:end_ix_val]

    

print("nr_samples train data:", nr_samples)

print("start_ix_val:", start_ix_val)

print("end_ix_val:", end_ix_val)
print("X:")

print(X.info())

print("*"*50)

print("X_test:")

print(X_test.info())

print("*"*50)

print("y:")

print(y.shape)
X.iloc[0:5,:]
y.iloc[0:5]
fig, axs = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(10,6))

axs = axs.flatten()

for i in range(0,5):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[i].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[i].set_title(y[i])

plt.tight_layout()    
y.value_counts()
fig, ax = plt.subplots(figsize=(8,5))

g = sns.countplot(y)
li_idxs = []

for i in range(10):

    for nr in range(10):

        ix = y[y==nr].index[i]

        li_idxs.append(ix) 
fig, axs = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(10,12))

axs = axs.flatten()

for n, i in enumerate(li_idxs):

    im = X.iloc[i]

    im = im.values.reshape(-1,28,28,1)

    axs[n].imshow(im[0,:,:,0], cmap=plt.get_cmap('gray'))

    axs[n].set_title(y[i])

plt.tight_layout()    
from sklearn.linear_model import Perceptron

clf_Perceptron = Perceptron(random_state=0)

param_grid = { 'penalty': ['l1','l2'], 'tol': [0.05, 0.1] }

GridCV_Perceptron = GridSearchCV(clf_Perceptron, param_grid, verbose=1, cv=5)

GridCV_Perceptron.fit(X_train,y_train)

score_grid_Perceptron = get_best_score(GridCV_Perceptron)
pred_val_perc = GridCV_Perceptron.predict(X_val)
acc_perc = print_validation_report(y_val, pred_val_perc)
plot_confusion_matrix(y_val, pred_val_perc)
from sklearn.linear_model import LogisticRegression

clf_LR = LogisticRegression(random_state=0)

param_grid = {'C': [0.014,0.012], 'multi_class': ['multinomial'],  

              'penalty': ['l1'],'solver': ['saga'], 'tol': [0.1] }

GridCV_LR = GridSearchCV(clf_LR, param_grid, verbose=1, cv=5)

GridCV_LR.fit(X_train,y_train)

score_grid_LR = get_best_score(GridCV_LR)
pred_val_lr = GridCV_LR.predict(X_val)

acc_lr = print_validation_report(y_val, pred_val_lr)
plot_confusion_matrix(y_val, pred_val_lr)
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=10)

clf_knn.fit(X_train,y_train)
pred_val_knn = clf_knn.predict(X_val)

acc_knn = print_validation_report(y_val, pred_val_knn)
plot_confusion_matrix(y_val, pred_val_knn)
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(random_state=0)

param_grid = {'max_depth': [15], 'max_features': [100],  

              'min_samples_split': [5],'n_estimators' : [50] }

GridCV_RF = GridSearchCV(clf_RF, param_grid, verbose=1, cv=5)

GridCV_RF.fit(X_train,y_train)

score_grid_RF = get_best_score(GridCV_RF)
pred_val_rf = GridCV_RF.predict(X_val)
acc_rf = print_validation_report(y_val, pred_val_rf)
plot_confusion_matrix(y_val, pred_val_rf)
from sklearn.svm import SVC

clf_svm = SVC(C=5, gamma=0.05, kernel='rbf', random_state=0)

clf_svm.fit(X_train,y_train)
pred_val_svm = clf_svm.predict(X_val)

acc_svm = print_validation_report(y_val, pred_val_svm)
plot_confusion_matrix(y_val, pred_val_svm)
batchsize = int(nr_samples/15) 
from sklearn.neural_network import MLPClassifier



clf_mlp = MLPClassifier(activation = "logistic", hidden_layer_sizes=(200,), random_state=0)

param_grid = { 'batch_size' : [batchsize] , 'max_iter': [600], 'alpha': [1e-4], 

               'solver': ['sgd'], 'learning_rate_init': [0.05,0.06],'tol': [1e-4] }

    

GridCV_MLP = GridSearchCV(clf_mlp, param_grid, verbose=1, cv=3)

GridCV_MLP.fit(X_train,y_train)

score_grid_MLP = get_best_score(GridCV_MLP)   
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(GridCV_MLP.best_estimator_.loss_curve_)



plt.xlabel("number of steps") 

plt.ylabel("Loss During GD")

plt.title("loss function")

plt.show()
pred_val_mlp = GridCV_MLP.predict(X_val)
acc_mlp = print_validation_report(y_val, pred_val_mlp)
plot_confusion_matrix(y_val, pred_val_mlp)
y_train = to_categorical(y_train, 10)

y_val_10 = to_categorical(y_val, 10)
def dense_model_0():

    model = Sequential()

    model.add(Dense(10, input_dim=784, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model_dense_0 = dense_model_0()

model_dense_0.summary()
model_dense_0.fit(X_train, y_train, epochs=50, batch_size=batchsize)
pred_val_dense0 = model_dense_0.predict_classes(X_val)
acc_fc0 = print_validation_report(y_val, pred_val_dense0)
plot_confusion_matrix(y_val, pred_val_dense0)
def dense_model_1():

    model = Sequential()

    model.add(Dense(100, input_dim=784, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model_dense_1 = dense_model_1()

model_dense_1.summary()
history_dense_1 = model_dense_1.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                    epochs=50, batch_size=batchsize)
plot_history_loss_and_acc(history_dense_1)
pred_val_dense1 = model_dense_1.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_dense1)

print(classification_report(y_val, pred_val_dense1))

acc_fc1 = accuracy_score(y_val, pred_val_dense1)

print(acc_fc1)
def dense_model_2():

    model = Sequential()

    model.add(Dense(100, input_dim=784, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model_dense_2 = dense_model_2()

model_dense_2.summary()
history_dense_2 = model_dense_2.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                    epochs=50, batch_size=batchsize)
plot_history_loss_and_acc(history_dense_2)
pred_val_dense2 = model_dense_2.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_dense2)

print(classification_report(y_val, pred_val_dense2))

acc_fc2 = accuracy_score(y_val, pred_val_dense2)

print(acc_fc2)
def dense_model_3():

    

    model = Sequential()  

    model.add(Dense(100, activation='relu', input_dim=784))

    model.add(Dense(200, activation='relu')) 

    model.add(Dense(100, activation='relu')) 

    model.add(Dense(10, activation='softmax'))

         

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    #model.compile(optimizer=RMSprop(lr=0.001),

    #         loss='categorical_crossentropy',

    #         metrics=['accuracy'])

    

    return model
model_dense_3 = dense_model_3()

model_dense_3.summary()
history_dense_3 = model_dense_3.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                    epochs=50, batch_size=batchsize)
plot_history_loss_and_acc(history_dense_3)
pred_val_dense3 = model_dense_3.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_dense3)

print(classification_report(y_val, pred_val_dense3))

acc_fc3 = accuracy_score(y_val, pred_val_dense3)

print(acc_fc3)
X_train.shape
X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_val = X_val.values.reshape(X_val.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)
X_train.shape
y_train.shape
batchsize = 128

epochs = 12
activation = 'relu'

adadelta = Adadelta()

loss = categorical_crossentropy
def cnn_model_1(activation):

    

    model = Sequential()

    

    model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=input_shape)) 

    

    model.add(Conv2D(64, (3, 3), activation=activation))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

 

    model.add(Flatten())



    model.add(Dense(128, activation=activation))

    model.add(Dropout(0.5))



    model.add(Dense(10, activation='softmax'))



    model.compile(loss=loss, optimizer=adadelta, metrics=['accuracy'])



    return model
model_cnn_1 = cnn_model_1(activation)

model_cnn_1.summary()
#model_cnn_1.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, verbose=1)

history_cnn_1 = model_cnn_1.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                   epochs=epochs, batch_size=batchsize, verbose=1)
plot_history_loss_and_acc(history_cnn_1)
pred_val_cnn1 = model_cnn_1.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_cnn1)

print(classification_report(y_val, pred_val_cnn1))

acc_cnn1 = accuracy_score(y_val, pred_val_cnn1)

print(acc_cnn1)

batch_size=90

epochs=30

def cnn_model_2(optimizer,loss):



    model = Sequential()



    model.add(Conv2D(32, (3, 3), padding = 'Same', activation="relu", input_shape=input_shape ))

    model.add(MaxPooling2D(pool_size = (2, 2)))



    model.add(Conv2D(32, (3, 3), activation="relu"))

    model.add(MaxPooling2D(pool_size = (2, 2)))



    model.add(Flatten())



    model.add(Dense(256, activation=activation))

    model.add(Dense(10, activation='softmax'))



    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy']) 



    return model
model_cnn_2 = cnn_model_2(adadelta, categorical_crossentropy)

model_cnn_2.summary()
#model_cnn_2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

history_cnn_2 = model_cnn_2.fit(X_train, y_train, validation_data=(X_val,y_val_10), 

                                epochs=epochs, batch_size=batchsize, verbose=1)
plot_history_loss_and_acc(history_cnn_2)
pred_val_cnn2 = model_cnn_2.predict_classes(X_val)

plot_confusion_matrix(y_val, pred_val_cnn2)

print(classification_report(y_val, pred_val_cnn2))

acc_cnn2 = accuracy_score(y_val, pred_val_cnn2)

print(acc_cnn2)
sample_submission = pd.read_csv('../input/sample_submission.csv')

if mode == "edit" :

    X = X[:nr_samples//2]

    y = y[:nr_samples//2]

    X_test = X_test[:nr_samples//2]

    sample_submission = sample_submission[:nr_samples//2]
print(X.shape)

print(y.shape)

print(X_test.shape)
print(GridCV_Perceptron.best_params_)

GridCV_Perceptron.best_estimator_.fit(X,y)
pred_test_perc = GridCV_Perceptron.best_estimator_.predict(X_test)

result_perc = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_perc})

result_perc.to_csv("subm_perc.csv",index=False)
print(GridCV_LR.best_params_)

GridCV_LR.best_estimator_.fit(X,y)
pred_test_lr = GridCV_LR.best_estimator_.predict(X_test)

result_lr = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_lr})

result_lr.to_csv("subm_lr.csv",index=False)
clf_knn.fit(X,y)
pred_test_knn = clf_knn.predict(X_test)

result_knn = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_knn})

result_knn.to_csv("subm_knn.csv",index=False)
print(GridCV_RF.best_params_)

GridCV_RF.best_estimator_.fit(X,y)
pred_test_rf = GridCV_RF.best_estimator_.predict(X_test)

result_rf = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_rf})

result_rf.to_csv("subm_rf.csv",index=False)
clf_svm.fit(X,y)
pred_test_svm = clf_svm.predict(X_test)

result_svm = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_svm})

result_svm.to_csv("subm_svm.csv",index=False)
print(GridCV_MLP.best_params_)

GridCV_MLP.best_estimator_.fit(X,y)
pred_test_mlp = GridCV_MLP.best_estimator_.predict(X_test)

result_mlp = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_mlp})

result_mlp.to_csv("subm_mlp.csv",index=False)
y = to_categorical(y, 10)
model_dense_1.fit(X,y)

pred_test_fc1 = model_dense_1.predict_classes(X_test)

result_fc1 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_fc1})

result_fc1.to_csv("dense_1.csv",index=False)
model_dense_2.fit(X,y)

pred_test_fc2 = model_dense_2.predict_classes(X_test)

result_fc2 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_fc2})

result_fc2.to_csv("dense_2.csv",index=False)
model_dense_3.fit(X,y)

pred_test_fc3 = model_dense_3.predict_classes(X_test)

result_fc3 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_fc3})

result_fc3.to_csv("dense_3.csv",index=False)
X = X.values.reshape(X.shape[0], img_rows, img_cols, 1)

X_test = X_test.values.reshape(X_test.shape[0], img_rows, img_cols, 1)

#y = to_categorical(y, 10)
batchsize = 128

epochs = 12

model_cnn_1 = cnn_model_1('relu')

model_cnn_1.fit(X, y, epochs=epochs, batch_size=batchsize, verbose=0)
pred_test_cnn_1 = model_cnn_1.predict(X_test)

pred_test_cnn_1 = np.argmax(pred_test_cnn_1,axis=1)

result_cnn_1 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_cnn_1})

result_cnn_1.to_csv("subm_cnn_1.csv",index=False)
batch_size=90

epochs=30

model_cnn_2 = cnn_model_2(adadelta, categorical_crossentropy)

model_cnn_2.fit(X, y, epochs=epochs, batch_size=batchsize, verbose=0)
pred_test_cnn_2 = model_cnn_2.predict(X_test)

pred_test_cnn_2 = np.argmax(pred_test_cnn_2,axis=1)

result_cnn_2 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_cnn_2})

result_cnn_2.to_csv("subm_cnn_2_adadelta.csv",index=False)
predictions = {'PERC': pred_test_perc, 'LR': pred_test_lr, 'KNN': pred_test_knn, 

               'RF': pred_test_rf, 'SVM': pred_test_svm, 'MLP': pred_test_mlp, 

               'DENSE1': pred_test_fc1, 'DENSE2': pred_test_fc2, 'DENSE3': pred_test_fc3, 

               'CNN1': pred_test_cnn_1, 'CNN2': pred_test_cnn_2}

df_predictions = pd.DataFrame(data=predictions) 

df_predictions.corr()
list_classifiers = ['PERC','LR','KNN','RF','SVM',

                    'MLP','DENSE1','DENSE2','DENSE3',

                    'CNN1','CNN2']
val_scores = [acc_perc, acc_lr, acc_knn, acc_rf, 

               acc_svm, acc_mlp, acc_fc1, acc_fc2, 

               acc_fc3, acc_cnn1, acc_cnn2]
score_perc  = 0.88057

score_lr    = 0.88700

score_knn   = 0.96557

score_rf    = 0.96028

score_svm   = 0.98100

score_mlp   = 0.96985



score_dns_1  = 0.95971 

score_dns_2  = 0.96228      

score_dns_3  = 0.96128

score_cnn_1  = 0.98928

score_cnn_2  = 0.99028
test_scores = [score_perc, score_lr, score_knn, score_rf, score_svm, score_mlp,

               score_dns_1, score_dns_2, score_dns_3, score_cnn_1, score_cnn_2]
trace1 = go.Scatter(x = list_classifiers, y = val_scores,

                   name="Validation", text = list_classifiers)

trace2 = go.Scatter(x = list_classifiers, y = test_scores,

                   name="Submission", text = list_classifiers)



data = [trace1, trace2]



layout = dict(title = "Validation and Submission Scores", 

              xaxis=dict(ticklen=10, zeroline= False),

              yaxis=dict(title = "Accuracy", side='left', ticklen=10,),                                  

              legend=dict(orientation="v", x=1.05, y=1.0),

              autosize=False, width=750, height=500,

              )



fig = dict(data = data, layout = layout)

iplot(fig)
model_cnn_2.optimizer
model_cnn_2_rmsprop = cnn_model_2(RMSprop(), categorical_crossentropy)

model_cnn_2_rmsprop.optimizer
model_cnn_2.fit(X, y, epochs=epochs, batch_size=batchsize, verbose=0)

pred_test_cnn_2 = model_cnn_2.predict(X_test)

pred_test_cnn_2 = np.argmax(pred_test_cnn_2,axis=1)

result_cnn_2 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_cnn_2})

result_cnn_2.to_csv("subm_cnn_2_rmsprop.csv",index=False)
model_cnn_2_adam = cnn_model_2(Adam(), categorical_crossentropy)

model_cnn_2_adam.optimizer
model_cnn_2.fit(X, y, epochs=epochs, batch_size=batchsize, verbose=0)

pred_test_cnn_2 = model_cnn_2.predict(X_test)

pred_test_cnn_2 = np.argmax(pred_test_cnn_2,axis=1)

result_cnn_2 = pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':pred_test_cnn_2})

result_cnn_2.to_csv("subm_cnn_2_adam.csv",index=False)
arr_y_val = y_val.values

false_cnn2 = pred_val_cnn2 != arr_y_val
fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10,12))

axs = axs.flatten()

for i, n  in enumerate(false_cnn2[:25]):

    im = X_val[false_cnn2][i,:,:,0]

    axs[i].imshow(im, cmap=plt.get_cmap('gray'))

    title = ("predicted: " + str(pred_val_cnn2[false_cnn2][i]) + 

            "\n" + "true: " + str(arr_y_val[false_cnn2][i]) )

    axs[i].set_title(title)

plt.tight_layout()    