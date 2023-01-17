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
from matplotlib import pyplot as plt 

import seaborn as sns

from keras.backend import clear_session

from keras.utils import to_categorical

from keras.layers import Dense

from keras.models import Sequential

from keras.metrics import AUC

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, classification_report

from tensorflow.random import set_seed

from tensorflow import get_logger





get_logger().setLevel('ERROR')



# We use these random seeds to ensure reproductibility

np.random.seed(1)

set_seed(1)
df = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')



df.info()



df.head(n=5)
%matplotlib inline

df.hist(bins=20, figsize=(20, 15))

plt.show()
#Let's remove the "histogram_tendency", standardize the data and split it between a train and test set

df = df.drop(['histogram_tendency'], axis=1)



X, y = df.drop(['fetal_health'],axis=1), df['fetal_health']



scaler = StandardScaler()

X = scaler.fit_transform(X)



encoder = OneHotEncoder()

y = encoder.fit_transform(y.values.reshape(-1,1)).toarray()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state = 42)



def compile_model():

    clear_session()



    model = Sequential()



    model.add(Dense(20, input_shape=(20,), activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[AUC(multi_label=True)])

    return model





loss, val_loss = [[],[]]



model = compile_model()

our_model = model.fit(X_train, y_train, validation_data=(X_test, y_test), 

                      epochs=100, verbose=0)



loss.append(our_model.history['loss'])

val_loss.append(our_model.history['val_loss'])



fig, ax = plt.subplots(figsize=(5,5))



ax.plot(loss[0], color='k', label='Train set')

ax.plot(val_loss[0], color='r', label='Test set')

ax.set_xlabel('Number of epochs')

ax.set_ylabel('Loss')

ax.set_xlim(0,30)

ax.legend()

model = compile_model()

our_model = model.fit(X_train, y_train, epochs=4, verbose=0)

y_pred0 = model.predict(X_test)



def print_f1score(model, X_train, y_train, y_test, y_pred):

    

    training_score = f1_score(np.argmax(y_train,axis=1), 

                              np.argmax(model.predict(X_train),axis=1), average='micro')



    test_score = f1_score(np.argmax(y_test,axis=1), 

                          np.argmax(y_pred,axis=1), average='micro')



    print('f1-score on the training set: %s'%training_score)

    print('f1-score on the test set: %s'%test_score)



print_f1score(model, X_train, y_train, y_test, y_pred0)

# Check out these nice tricks of Dennis T to plot confusion matrix

# [https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea]



def print_confusion_matrix(model, X_train, y_train, y_test, y_pred):

    train_confusion = confusion_matrix(np.argmax(y_train,axis=1), np.argmax(model.predict(X_train),axis=1))

    test_confusion = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))



    fig, ax = plt.subplots(1,2,figsize=(12,5))



    sns.heatmap(train_confusion/np.sum(train_confusion), ax=ax[0], annot=True, fmt='.2%', cmap='Reds')

    ax[0].set_xlabel('Predicted labels')

    ax[0].set_ylabel('Actual labels')

    ax[0].set_title('Confusion matrix (train set)')



    sns.heatmap(test_confusion/np.sum(test_confusion), ax=ax[1], annot=True, fmt='.2%', cmap='Reds')

    ax[1].set_title('Confusion matrix (test set)')

    ax[1].set_xlabel('Predicted labels')

    ax[1].set_ylabel('Actual labels')



print_confusion_matrix(model, X_train, y_train, y_test, y_pred0)

# Suspect correctly labeled

a = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred0, axis=1))

tmp1 = np.round(100*a[1,1]/np.sum(a[1,:]),1)



# Pathological correctly labeled

tmp2 = np.round(100*a[2,2]/np.sum(a[2,:]),1)



print('The confusion matrix show us that, on unseen data:')

print(tmp1,'% of the Suspect foetus are correctly labeled')

print(tmp2,'% of the Pathological foetus are correctly labeled')
def my_new_model(act, opt):

    clear_session()

    model = Sequential()

    model.add(Dense(20, input_shape=(20,), activation=act))

    model.add(Dense(20, activation=act))

    model.add(Dense(20, activation=act))

    model.add(Dense(20, activation=act))

    model.add(Dense(20, activation=act))

    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model



new_model = KerasClassifier(build_fn=my_new_model, verbose=0)



parameters = dict(opt = ['adam', 'sgd', 'adamax'], 

                  act=['relu', 'softmax', 'tanh', 'selu'],  

                  batch_size=[32, 64, 128, 256, 512])



random_search = RandomizedSearchCV(new_model, param_distributions=parameters, 

                                   n_iter=30, scoring='roc_auc', random_state=123)



res = random_search.fit(X_train, y_train)



print(res.best_params_)
#Let's see how we perform with the prescribed optimizer and activation function! 



optimized_model = my_new_model('relu', 'adam')



optimized_model.fit(X_train, y_train, epochs=14, batch_size=32, verbose=0)



y_pred = optimized_model.predict(X_test)



print_f1score(optimized_model, X_train, y_train, y_test, y_pred)
print_confusion_matrix(optimized_model, X_train, y_train, y_test, y_pred)
# Suspect correctly labeled

a = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

tmp1 = np.round(100*a[1,1]/np.sum(a[1,:]),1)



# Pathological correctly labeled

tmp2 = np.round(100*a[2,2]/np.sum(a[2,:]),1)



print('Thanks to the hyperparameter tuning, on unseen data, we have now:')

print(tmp1,'% of the Suspect foetus are correctly labeled')

print(tmp2,'% of the Pathological foetus are correctly labeled')
precision = dict()

recall = dict()

precision_opt = dict()

recall_opt = dict()



for i in range(3):

    precision[i], recall[i], _ = precision_recall_curve(y_test[:,i],y_pred0[:,i])

    precision_opt[i], recall_opt[i], _ = precision_recall_curve(y_test[:,i],y_pred[:,i])



fig, ax = plt.subplots()



colors = ['k','r','b']

labels = ['Healthy','Suspect','Pathological']



for i in range(3):

    ax.plot(recall[i],precision[i],color=colors[i],label=labels[i],linestyle='--')

    ax.plot(recall_opt[i],precision_opt[i],color=colors[i],label=labels[i],linestyle='-')



ax.set_xlabel('Recall TP/(FN+TP)')

ax.set_ylabel('Precision TP/(FP+TP)')



plt.legend()

plt.show()