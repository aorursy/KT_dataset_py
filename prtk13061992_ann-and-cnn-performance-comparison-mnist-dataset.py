#install required libraries
import pandas as pd
import numpy as np

#data visualization packages
import matplotlib.pyplot as plt

#keras packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

#model evaluation packages
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, cross_validate

#other packages
import time as time
from IPython.display import display, Markdown
from IPython.display import display
from time import sleep
from IPython.display import Markdown as md
#read mnist fashion dataset
mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#reshape data from 3-D to 2-D array
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
#feature scaling
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

#fit and transform training dataset
X_train = minmax.fit_transform(X_train)

#transform testing dataset
X_test = minmax.transform(X_test)
print('Number of unique classes: ', len(np.unique(y_train)))
print('Classes: ', np.unique(y_train))
fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))          #create subplot
ax = axes.ravel()
for i in range(10):
    ax[i].imshow(X_train[i].reshape(28,28))                        #print image
    ax[i].title.set_text('Class: ' + str(y_train[i]))              #print class
plt.subplots_adjust(hspace=0.5)                                    #increase horizontal space
plt.show()                                                         #display image
#initializing CNN model
classifier_e25 = Sequential()

#add 1st hidden layer
classifier_e25.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))

#add output layer
classifier_e25.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
#compile the neural network
classifier_e25.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model summary
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
classifier_e25.summary()
#include time details
dh = display('',display_id=True)
dh.update(md("<br>Training is in progress....."))
t1 = time.time()

#fit training dataset into the model
classifier_e25_fit = classifier_e25.fit(X_train, y_train, epochs=25, verbose=0)
tt = time.time()-t1
dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))

display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

#plot the graphs
#accuracy graph
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
ax = axes.ravel()
ax[0].plot(range(0,classifier_e25_fit.params['epochs']), [acc * 100 for acc in classifier_e25_fit.history['accuracy']], label='Accuracy')
ax[0].set_title('Accuracy vs. epoch', fontsize=15)
ax[0].set_ylabel('Accuracy', fontsize=15)
ax[0].set_xlabel('epoch', fontsize=15)
ax[0].legend()

#losso graph
ax[1].plot(range(0,classifier_e25_fit.params['epochs']), classifier_e25_fit.history['loss'], label='Loss', color='r')
ax[1].set_title('Loss vs. epoch', fontsize=15)
ax[1].set_ylabel('Loss', fontsize=15)
ax[1].set_xlabel('epoch', fontsize=15)
ax[1].legend()

#display the graph
plt.show()
#include timing information
dh = display('',display_id=True)
dh.update(md("<br>Model evaluation is in progress..."))
t2 = time.time()

#evaluate the model for testing dataset
test_loss_e25 = classifier_e25.evaluate(X_test, y_test, verbose=0)
et = time.time()-t2
dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))

display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Model Evaluation Summary \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

#calculate evaluation parameters
f1_e25 = f1_score(y_test, classifier_e25.predict_classes(X_test), average='micro')
roc_e25 = roc_auc_score(y_test, classifier_e25.predict_proba(X_test), multi_class='ovo')

#create evaluation dataframe
stats_e25 = pd.DataFrame({'Test accuracy' : round(test_loss_e25[1]*100,3),
                      'F1 score'      : round(f1_e25,3),
                      'ROC AUC score' : round(roc_e25,3),
                      'Total Loss'    : round(test_loss_e25[0],3)}, index=[0])

#print evaluation dataframe
display(stats_e25)
def model_cv(epoch, cv):
    '''Function for cross validation'''
    
    #Model Initializing, Compiling and Fitting
    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return classifier

    #model summary
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
    build_classifier().summary()

    #create KerasClassifier object
    classifier_cv = KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=epoch, verbose=0)
    scoring = {'acc' : 'accuracy',
                    'f1'  : 'f1_micro',
                    'roc' : 'roc_auc_ovo',
                    'loss': 'neg_log_loss'}

    #include timing information
    dh = display('',display_id=True)
    dh.update(md("<br>Training is in progress....."))
    t1 = time.time()
    
    #perform cross validation
    scores = cross_validate(classifier_cv, X_train, y_train, cv=cv, scoring=scoring, verbose=0, return_train_score=True)
    tt = time.time()-t1
    dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

    #plot graphs
    #accuracy graph
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(range(1,len(scores['train_acc'])+1), [acc * 100 for acc in scores['train_acc']], label='Accuracy')
    ax[0].set_title('Accuracy vs. Cross Validation', fontsize=15)
    ax[0].set_ylabel('Accuracy', fontsize=15)
    ax[0].set_xlabel('Cross Validation', fontsize=15)
    ax[0].legend()

    #loss graph
    ax[1].plot(range(1,len(scores['train_loss'])+1), np.abs(scores['train_loss']), label='Loss', color='r')
    ax[1].set_title('Loss vs. Cross Validation', fontsize=15)
    ax[1].set_ylabel('Loss', fontsize=15)
    ax[1].set_xlabel('Cross Validation', fontsize=15)
    ax[1].legend()

    #display the graph
    plt.show()


    #Evaluating the model
    dh = display('',display_id=True)
    dh.update(md("<br><br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(np.sum(scores['score_time']),3))))

    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* Model Evaluation \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))
    
    #create the model evaluation dataframe
    stats = pd.DataFrame({'Test accuracy' : round(np.mean(scores['test_acc'])*100,3),
                          'F1 score'      : round(np.mean(scores['test_f1']),3), 
                          'ROC AUC score' : round(np.mean(scores['test_roc']),3),
                          'Total Loss'    : round(np.abs(np.mean(scores['test_loss'])),3)}, index=[0])
    #print the dataframe
    display(stats)

    #return the classifier and evaluation parameter details
    return scores, stats
#run the model for 5-Fold cross validation
scores_5cv, stats_5cv = model_cv(epoch=25, cv=5)
#run the model for 10-Fold cross validation
scores_10cv, stats_10cv = model_cv(epoch=25, cv=10)
def add_value_labels(ax, spacing=5):
    '''add label details on each bar graph'''
    
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
#Plot the graph
x_axis = ['cv=1', 'cv=5', 'cv=10']
y_axis = [classifier_e25_fit.history['accuracy'][-1]*100, np.mean(scores_5cv['train_acc']*100), np.mean(scores_10cv['train_acc']*100)]

#create series with y_axis values
freq_series = pd.Series(y_axis)

display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))                                    #figure size
ax = freq_series.plot(kind='bar')                             #plot the type of graph
plt.xlabel('Number of Cross Validation', fontsize=15)         #xlabel
plt.ylabel('Training Accuracy', fontsize=15)                  #ylabel
plt.ylim(min(y_axis)-1,max(y_axis)+1)                         #limit the y_axis dynamically
plt.title('Bar graph for Number of Cross Validation vs. Training Accuracy', fontsize=15)    #title
ax.set_xticklabels(x_axis)                                    #x-ticks

# Put labels on each bar graph
add_value_labels(ax)  
#Plot the graph
x_axis = ['cv=1', 'cv=5', 'cv=10']
y_axis = [stats_e25['Test accuracy'][0], stats_5cv['Test accuracy'][0], stats_10cv['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of Cross Validation', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of Cross Validation vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

# Put labels on each bar graph
add_value_labels(ax)
def model_epcoh(epoch):
    '''Function to run Neural Network for different epochs'''
    
    #Model Initializing, Compiling and Fitting
    classifier = Sequential()
    classifier.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
    classifier.summary()

    #include timing details
    dh = display('',display_id=True)
    dh.update(md("<br>Training is in progress....."))
    t1 = time.time()
    #fit the model with training dataset
    classifier_fit = classifier.fit(X_train, y_train, epochs=epoch, verbose=0)
    tt = time.time()-t1
    dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))

    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

    #plot the graph
    #accuracy graph
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(range(0,classifier_fit.params['epochs']), [acc * 100 for acc in classifier_fit.history['accuracy']], label='Accuracy')
    ax[0].set_title('Accuracy vs. epoch', fontsize=15)
    ax[0].set_ylabel('Accuracy', fontsize=15)
    ax[0].set_xlabel('epoch', fontsize=15)
    ax[0].legend()

    #loss graph
    ax[1].plot(range(0,classifier_fit.params['epochs']), classifier_fit.history['loss'], label='Loss', color='r')
    ax[1].set_title('Loss vs. epoch', fontsize=15)
    ax[1].set_ylabel('Loss', fontsize=15)
    ax[1].set_xlabel('epoch', fontsize=15)
    ax[1].legend()

    #display the graph
    plt.show()

    #Evaluating the model    
    dh = display('',display_id=True)
    dh.update(md("<br>Model evaluation is in progress..."))
    t2 = time.time()
    
    #model evaluation
    test_loss = classifier.evaluate(X_test, y_test, verbose=0)
    et = time.time()-t2
    dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Model Evaluation Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

    #calculate the evaluation parameter
    f1 = f1_score(y_test, classifier.predict_classes(X_test), average='micro')
    roc = roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovo')

    #create the model evaluation dataframe
    stats = pd.DataFrame({'Test accuracy' : round(test_loss[1]*100,3),
                          'F1 score'      : round(f1,3),
                          'ROC AUC score' : round(roc,3),
                          'Total Loss'    : round(test_loss[0],3)}, index=[0])
    
    #print the dataframe
    display(stats)
    
    #return the classifier and model evaluation details
    return classifier_fit, stats
#run the model for 50 epochs
classifier_e50, stats_e50 = model_epcoh(50)
#run the model for 100 epochs
classifier_e100, stats_e100 = model_epcoh(100)
#run the model for 200 epochs
classifier_e200, stats_e200 = model_epcoh(200)
#Plot the graph
x_axis = ['epoch=25', 'epoch=50', 'epoch=100', 'epoch=200']
y_axis = [classifier_e25_fit.history['accuracy'][-1]*100, classifier_e50.history['accuracy'][-1]*100, classifier_e100.history['accuracy'][-1]*100, classifier_e200.history['accuracy'][-1]*100]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of Epochs', fontsize=15)
plt.ylabel('Training Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of Epochs vs. Training Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

# add labels for each bar graph
add_value_labels(ax)
#Plot the graph
x_axis = ['epoch=25', 'epoch=50', 'epoch=100', 'epoch=200']
y_axis = [stats_e25['Test accuracy'][0], stats_e50['Test accuracy'][0], stats_e100['Test accuracy'][0], stats_e200['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of Epochs', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of Epochs vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add labels for each bar graph
add_value_labels(ax)
#Model Initializing, Compiling and Fitting
classifier_2dl = Sequential()
classifier_2dl.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
classifier_2dl.add(Dense(units = 128, kernel_initializer='uniform', activation='relu'))
classifier_2dl.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
classifier_2dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
classifier_2dl.summary()

#include timing details
dh = display('',display_id=True)
dh.update(md("<br>Training is in progress....."))
t1 = time.time()

#fit the model with training dataset
classifier_2dl_fit = classifier_2dl.fit(X_train, y_train, epochs=50, verbose=0)  #batch_size=32 (default)
tt = time.time()-t1
dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

#plot the graph
#accuracy graph
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
ax = axes.ravel()
ax[0].plot(range(0,classifier_2dl_fit.params['epochs']), [acc * 100 for acc in classifier_2dl_fit.history['accuracy']], label='Accuracy')
ax[0].set_title('Accuracy vs. epoch', fontsize=15)
ax[0].set_ylabel('Accuracy', fontsize=15)
ax[0].set_xlabel('epoch', fontsize=15)
ax[0].legend()

#loss graph
ax[1].plot(range(0,classifier_2dl_fit.params['epochs']), classifier_2dl_fit.history['loss'], label='Loss', color='r')
ax[1].set_title('Loss vs. epoch', fontsize=15)
ax[1].set_ylabel('Loss', fontsize=15)
ax[1].set_xlabel('epoch', fontsize=15)
ax[1].legend()

#display the graph
plt.show()

#Evaluating the model
dh = display('',display_id=True)
dh.update(md("<br>Model evaluation is in progress..."))
t2 = time.time()

#model evaluation
test_loss_2dl = classifier_2dl.evaluate(X_test, y_test, verbose=0)
et = time.time()-t2
dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Model Evaluation Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

#calculate the model evaluation parameter
f1_2dl = f1_score(y_test, classifier_2dl.predict_classes(X_test), average='micro')
roc_2dl = roc_auc_score(y_test, classifier_2dl.predict_proba(X_test), multi_class='ovo')

#create the model evaluation dataframe
stats_2dl = pd.DataFrame({'Test accuracy' : round(test_loss_2dl[1]*100,3),
                      'F1 score'      : round(f1_2dl,3),
                      'ROC AUC score' : round(roc_2dl,3),
                      'Total Loss'    : round(test_loss_2dl[0],3)}, index=[0])

#print the dataframe
display(stats_2dl)
#Model Initializing, Compiling and Fitting
classifier_3dl = Sequential()
classifier_3dl.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
classifier_3dl.add(Dense(units = 128, kernel_initializer='uniform', activation='relu'))
classifier_3dl.add(Dense(units = 256, kernel_initializer='uniform', activation='relu'))
classifier_3dl.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
classifier_3dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
classifier_3dl.summary()

#include timing details
dh = display('',display_id=True)
dh.update(md("<br>Training is in progress....."))
t1 = time.time()

#fit the model with training dataset
classifier_3dl_fit = classifier_3dl.fit(X_train, y_train, epochs=50, verbose=0)  #batch_size=32 (default)
tt = time.time()-t1
dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

#plot the graph
#accuracy graph
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
ax = axes.ravel()
ax[0].plot(range(0,classifier_3dl_fit.params['epochs']), [acc * 100 for acc in classifier_3dl_fit.history['accuracy']], label='Accuracy')
ax[0].set_title('Accuracy vs. epoch', fontsize=15)
ax[0].set_ylabel('Accuracy', fontsize=15)
ax[0].set_xlabel('epoch', fontsize=15)
ax[0].legend()

#loss graph
ax[1].plot(range(0,classifier_3dl_fit.params['epochs']), classifier_3dl_fit.history['loss'], label='Loss', color='r')
ax[1].set_title('Loss vs. epoch', fontsize=15)
ax[1].set_ylabel('Loss', fontsize=15)
ax[1].set_xlabel('epoch', fontsize=15)
ax[1].legend()

#display the graph
plt.show()

#Evaluate the model
dh = display('',display_id=True)
dh.update(md("<br>Model evaluation is in progress..."))
t2 = time.time()

#model evaluation
test_loss_3dl = classifier_3dl.evaluate(X_test, y_test, verbose=0)
et = time.time()-t2
dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Model Evaluation Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

#calculate the model evaluation parameter
f1_3dl = f1_score(y_test, classifier_3dl.predict_classes(X_test), average='micro')
roc_3dl = roc_auc_score(y_test, classifier_3dl.predict_proba(X_test), multi_class='ovo')

#create the model evaluation dataframe
stats_3dl = pd.DataFrame({'Test accuracy' : round(test_loss_3dl[1]*100,3),
                      'F1 score'      : round(f1_3dl,3),
                      'ROC AUC score' : round(roc_3dl,3),
                      'Total Loss'    : round(test_loss_3dl[0],3)}, index=[0])

#print the dataframe
display(stats_3dl)
#Plot the graph
x_axis = ['1', '2', '3']
y_axis = [classifier_e50.history['accuracy'][-1]*100, classifier_2dl_fit.history['accuracy'][-1]*100, classifier_3dl_fit.history['accuracy'][-1]*100]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of Dense Layer', fontsize=15)
plt.ylabel('Training Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of Dense Layer vs. Training Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

# add labels for each bar graph
add_value_labels(ax)
#Plot the graph
x_axis = ['1', '2', '3']
y_axis = [stats_e50['Test accuracy'][0], stats_2dl['Test accuracy'][0], stats_3dl['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of Dense Layer', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of Dense Layer vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

# add labels for each bar graph
add_value_labels(ax)
def model_dropout(rate):
    '''Neural Network Model with Dropout'''
    
    #Model Initializing, Compiling and Fitting
    classifier = Sequential()
    classifier.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate = rate))
    classifier.add(Dense(units = 128, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate = rate))
    classifier.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model summary
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
    classifier.summary()

    #include timing details
    dh = display('',display_id=True)
    dh.update(md("<br>Training is in progress....."))
    t1 = time.time()
    
    #fit the model with training dataset
    classifier_fit = classifier.fit(X_train, y_train, epochs=50, verbose=0)
    tt = time.time()-t1
    dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

    #plot the graph
    #accuracy graph
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(range(0,classifier_fit.params['epochs']), [acc * 100 for acc in classifier_fit.history['accuracy']], label='Accuracy')
    ax[0].set_title('Accuracy vs. epoch', fontsize=15)
    ax[0].set_ylabel('Accuracy', fontsize=15)
    ax[0].set_xlabel('epoch', fontsize=15)
    ax[0].legend()

    #loss graph
    ax[1].plot(range(0,classifier_fit.params['epochs']), classifier_fit.history['loss'], label='Loss', color='r')
    ax[1].set_title('Loss vs. epoch', fontsize=15)
    ax[1].set_ylabel('Loss', fontsize=15)
    ax[1].set_xlabel('epoch', fontsize=15)
    ax[1].legend()

    #display the graph
    plt.show()

    #Evaluae the model
    dh = display('',display_id=True)
    dh.update(md("<br>Model evaluation is in progress..."))
    t2 = time.time()
    
    #model evaluation
    test_loss = classifier.evaluate(X_test, y_test, verbose=0)
    et = time.time()-t2
    dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Model Evaluation Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

    #calculate the model evaluation parameters
    f1 = f1_score(y_test, classifier.predict_classes(X_test), average='micro')
    roc = roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovo')

    #create model evaluation dataframe
    stats = pd.DataFrame({'Test accuracy' : round(test_loss[1]*100,3),
                          'F1 score'      : round(f1,3),
                          'ROC AUC score' : round(roc,3),
                          'Total Loss'    : round(test_loss[0],3)}, index=[0])
    
    #print the dataframe
    display(stats)
    
    #return the classifier and model evaluation details
    return classifier_fit, stats
#run the neural network model with dropout with rate=0.1
classifier_1d, stats_1d = model_dropout(0.1)
#run the neural network model with dropout with rate=0.2
classifier_2d, stats_2d = model_dropout(0.2)
#run the neural network model with dropout with rate=0.3
classifier_3d, stats_3d = model_dropout(0.3)
#Plot the model
x_axis = ['rate=0.0', 'rate=0.1', 'rate=0.2', 'rate=0.3']
y_axis = [classifier_e50.history['accuracy'][-1]*100, classifier_1d.history['accuracy'][-1]*100, classifier_2d.history['accuracy'][-1]*100, classifier_3d.history['accuracy'][-1]*100]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Dropout Rates', fontsize=15)
plt.ylabel('Training Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Dropout Rates vs. Training Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)
#Plotting
x_axis = ['rate=0.0', 'rate=0.1', 'rate=0.2', 'rate=0.3']
y_axis = [stats_e50['Test accuracy'][0], stats_1d['Test accuracy'][0], stats_2d['Test accuracy'][0], stats_3d['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Dropout Rates', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Dropout Rates vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)
#read mnist dataset
mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#reshape the dataframe
X_train=X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

#feature scaling
X_train=X_train / 255.0
X_test=X_test/255.0

#print the shape of each dataframe
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
def model_cnn(count=1):
    '''Convolution Neural Network'''
    
    #Model Initializing, Compiling and Fitting
    classifier = Sequential()
    
    #convolution layer
    classifier.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
    
    #max-pooling layer
    classifier.add(MaxPooling2D(2,2))
    
    #in case of multiple convolution layer
    if count>1:
        for i in range(count-1):
            classifier.add(Convolution2D(32, (3,3), activation='relu'))
            classifier.add(MaxPooling2D(2,2))
            
    #flatten layer
    classifier.add(Flatten())
    
    #fully connected layer
    #dense (hidden) layer
    classifier.add(Dense(units = 256, kernel_initializer='uniform', activation='relu'))
    
    #output layer
    classifier.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
    
    #compile the model
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model summary
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  Model  Summary  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))
    classifier.summary()

    #include timing details
    dh = display('',display_id=True)
    dh.update(md("<br>Training is in progress....."))
    t1 = time.time()
    
    #fit the model with training dataset
    classifier_fit = classifier.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    tt = time.time()-t1
    dh.update(md("<br>Training is completed! Total training time: **{} seconds**".format(round(tt,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*     Training Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br>'))

    #plot the graph
    #accuracy graph
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
    ax = axes.ravel()
    ax[0].plot(range(0,classifier_fit.params['epochs']), [acc * 100 for acc in classifier_fit.history['accuracy']], label='Accuracy')
    ax[0].set_title('Accuracy vs. epoch', fontsize=15)
    ax[0].set_ylabel('Accuracy', fontsize=15)
    ax[0].set_xlabel('epoch', fontsize=15)
    ax[0].legend()

    #loss graph
    ax[1].plot(range(0,classifier_fit.params['epochs']), classifier_fit.history['loss'], label='Loss', color='r')
    ax[1].set_title('Loss vs. epoch', fontsize=15)
    ax[1].set_ylabel('Loss', fontsize=15)
    ax[1].set_xlabel('epoch', fontsize=15)
    ax[1].legend()

    #display the graph
    plt.show()

    #Evaluate the model
    dh = display('',display_id=True)
    dh.update(md("<br>Model evaluation is in progress..."))
    t2 = time.time()
    
    #model evaluation
    test_loss = classifier.evaluate(X_test, y_test, verbose=0)
    et = time.time()-t2
    dh.update(md("<br>Model evaluation is completed! Total evaluation time: **{} seconds**".format(round(et,3))))
    display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Model Evaluation Summary    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***'))

    #calculate the model evaluation parameters
    f1 = f1_score(y_test, classifier.predict_classes(X_test), average='micro')
    roc = roc_auc_score(y_test, classifier.predict_proba(X_test), multi_class='ovo')

    #create model evaluation dtaaframe
    stats = pd.DataFrame({'Test accuracy' : round(test_loss[1]*100,3),
                          'F1 score'      : round(f1,3),
                          'ROC AUC score' : round(roc,3),
                          'Total Loss'    : round(test_loss[0],3)}, index=[0])
    
    #print the dataframe
    display(stats)
    
    #return the classifier and model evaluation details
    return classifier_fit, stats
#run the CNN model with 1 layer
classifier_1cnn, stats_1cnn = model_cnn(1)
#run the CNN model with 2 layer
classifier_2cnn, stats_2cnn = model_cnn(2)
#run the CNN model with 3 layer
classifier_3cnn, stats_3cnn = model_cnn(3)
#Plot the graph
x_axis = ['0', '1', '2', '3']
y_axis = [classifier_e50.history['accuracy'][-1]*100, classifier_1cnn.history['accuracy'][-1]*100, classifier_2cnn.history['accuracy'][-1]*100, classifier_3cnn.history['accuracy'][-1]*100]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of CNN Layer', fontsize=15)
plt.ylabel('Training Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of CNN Layer vs. Training Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)
#Plot the graph
x_axis = ['0', '1', '2', '3']
y_axis = [stats_e50['Test accuracy'][0], stats_1cnn['Test accuracy'][0], stats_2cnn['Test accuracy'][0], stats_3cnn['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Number of CNN Layer', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for Number of CNN Layer vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)
#Plot the graph
x_axis = ['cv=1 epoch=25', 
          'cv=5 epoch=25', 
          'cv=10 epoch=25', 
          'epoch=25', 
          'epoch=50', 
          'epoch=100', 
          'epoch=200', 
          '1 Dense Layer', 
          '2 Dense Layer', 
          '3 Dense Layer', 
          'Dropout rate=0.0', 
          'Dropout rate=0.1', 
          'Dropout rate=0.2', 
          'Dropout rate=0.3', 
          '0 CNN Layer', 
          '1 CNN Layer', 
          '2 CNN Layer', 
          '3 CNN Layer']
          
y_axis = [classifier_e25_fit.history['accuracy'][-1]*100, 
          np.mean(scores_5cv['train_acc']*100), 
          np.mean(scores_10cv['train_acc']*100),
          classifier_e25_fit.history['accuracy'][-1]*100, 
          classifier_e50.history['accuracy'][-1]*100, 
          classifier_e100.history['accuracy'][-1]*100, 
          classifier_e200.history['accuracy'][-1]*100,
          classifier_e50.history['accuracy'][-1]*100, 
          classifier_2dl_fit.history['accuracy'][-1]*100, 
          classifier_3dl_fit.history['accuracy'][-1]*100,
          classifier_e50.history['accuracy'][-1]*100, 
          classifier_1d.history['accuracy'][-1]*100, 
          classifier_2d.history['accuracy'][-1]*100, 
          classifier_3d.history['accuracy'][-1]*100,
          classifier_e50.history['accuracy'][-1]*100, 
          classifier_1cnn.history['accuracy'][-1]*100, 
          classifier_2cnn.history['accuracy'][-1]*100, 
          classifier_3cnn.history['accuracy'][-1]*100]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Models', fontsize=15)
plt.ylabel('Training Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for All Models vs. Training Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)
#Plot the model
x_axis = ['cv=1 epoch=25', 
          'cv=5 epoch=25', 
          'cv=10 epoch=25', 
          'epoch=25', 
          'epoch=50', 
          'epoch=100', 
          'epoch=200', 
          '1 Dense Layer', 
          '2 Dense Layer', 
          '3 Dense Layer', 
          'Dropout rate=0.0', 
          'Dropout rate=0.1', 
          'Dropout rate=0.2', 
          'Dropout rate=0.3', 
          '0 CNN Layer', 
          '1 CNN Layer', 
          '2 CNN Layer', 
          '3 CNN Layer']
          
y_axis = [stats_e25['Test accuracy'][0], 
          stats_5cv['Test accuracy'][0], 
          stats_10cv['Test accuracy'][0], 
          stats_e25['Test accuracy'][0], 
          stats_e50['Test accuracy'][0], 
          stats_e100['Test accuracy'][0], 
          stats_e200['Test accuracy'][0], 
          stats_e50['Test accuracy'][0], 
          stats_2dl['Test accuracy'][0], 
          stats_3dl['Test accuracy'][0], 
          stats_e50['Test accuracy'][0], 
          stats_1d['Test accuracy'][0], 
          stats_2d['Test accuracy'][0], 
          stats_3d['Test accuracy'][0], 
          stats_e50['Test accuracy'][0], 
          stats_1cnn['Test accuracy'][0], 
          stats_2cnn['Test accuracy'][0], 
          stats_3cnn['Test accuracy'][0]]

#create series with y_axis values
freq_series = pd.Series(y_axis)

#plot the graph
display(Markdown('<br>**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*    Graph Plot    \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***<br><br>'))
plt.figure(figsize=(12,6))
ax = freq_series.plot(kind='bar')
plt.xlabel('Models', fontsize=15)
plt.ylabel('Test Accuracy', fontsize=15)
plt.ylim(min(y_axis)-1,max(y_axis)+1)
plt.title('Bar graph for All Models vs. Test Accuracy', fontsize=15)
ax.set_xticklabels(x_axis)

#add label for each graph
add_value_labels(ax)