# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

#print(os.listdir())



#Plot Graph

import matplotlib.pyplot as plt



#--------------For Machine Learning-----------------#

##Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

##Modeling

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

#CNN

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import np_utils

##Model Saving and Reading

from sklearn.externals import joblib

from keras.models import load_model
#train_data = pd.read_csv('../input/train.csv')

#test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
#Show the training sample digit

plt.figure(num='training_sample',figsize=(10,10)) 

for row in range(1,26):

    plt.subplot(5, 5, row) #row must be >0 for subplot function

    plt.title('Digit:' + str(train_data.iloc[row-1,0]))

    plt.axis('off')

    plt.imshow(train_data.iloc[row-1,1:].values.reshape(28,28))
#Split tarin-test sets of 80%-20% respectively, 

#and set the random_state for consistent even I rerun the project.

x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:, 1:], 

                                                      train_data.iloc[:, 0], test_size = 0.2, random_state = 1)
#Data Description

x_train.describe()
y_train.value_counts().sort_index()
#Function to find out all scaling result to compare:

def model_score(model_name , x_train, x_test, y_train, y_test):

    model_path = '../input/digit-recognition-model-backup/' #read back the trained model

    score_save_path = '../input/'

    history_path = '../input/'

    score_list = {}

    scaling_method = {}

    model_algo = {}

    stdsc = StandardScaler()    

    scaling_method = {'original':[x_train, x_test],

                      'Standarisation':[stdsc.fit_transform(x_train), stdsc.fit_transform(x_test)],

                      'Mean Normalisation':[x_train.apply(lambda x: (x - np.mean(x))/(255-0)), 

                                           x_test.apply(lambda x: (x - np.mean(x))/(255-0))],

                      'Unit':[x_train.apply(lambda x: (x - 0)/(255 - 0)),

                                     x_test.apply(lambda x: (x - 0)/(255 - 0))]}

    for method in scaling_method:

        _x_train, _x_test = scaling_method[method]

        

        ###SVM###-----------------------------------------------------------

        if model_name.upper() == 'SVM':

            try:

                model = joblib.load(model_path + 'svm_model' + '_' + method + '.pkl')

                print('Model Reading Success')

            except: #If no exist model, we train

                print('No existed Model, it is fitting...')

                model = svm.SVC(gamma = 0.0001) #Since 'Auto' for orginial data is expensive

                model.fit(_x_train, y_train)

                print('Model is fitted')

            #Model Saving

            #joblib.dump(model, model_path + 'svm_model' + '_' + method + '.pkl') # Kaggle Only provide read mode

        ###Logistics###-----------------------------------------------------    

        elif model_name.upper() == 'LOGISTIC':

            try:

                model = joblib.load(model_path + 'log_model' + '_' + method + '.pkl')

                print('Model Reading Success')

            except:

                print('No existed Model, it is fitting...')

                model = LogisticRegression(random_state = 1)

                model.fit(_x_train, y_train)

                print('Model is fitted')

            #Model Saving

            #joblib.dump(model, model_path + 'log_model' + '_' + method + '.pkl')

        

        ###Decision Tree###-------------------------------------------------

        elif model_name.upper() == 'DECISION TREE':

            try:

                model = joblib.load(model_path + 'tree_model' + '_' + method + '.pkl')

                print('Model Reading Success')

            except:

                model = DecisionTreeClassifier(random_state = 1)

                model.fit(_x_train, y_train)

                print('Model is fitted')

            #Model Saving

            #joblib.dump(model, model_path + 'tree_model' + '_' + method + '.pkl')

        

        ###Random Forecast###-----------------------------------------------

        elif model_name.upper() == 'RANDOM FOREST':

            try:

                model = joblib.load(model_path + 'RF_model' + '_' + method + '.pkl')

                print('Model Reading Success')

            except:

                model = RandomForestClassifier(random_state = 1)

                model.fit(_x_train, y_train)

                print('Model is fitted')

            #Model Saving

            #joblib.dump(model, model_path + 'RF_model' + '_' + method + '.pkl')





        ###Convolutional Neural Network (CNN) ###----------------------------

        elif model_name.upper() == 'CNN':

            num_class = 10 #Digit 0-9

            #Data Reshaping

            if isinstance(_x_train, pd.DataFrame):

                re_x_train = _x_train.values.reshape(_x_train.shape[0], 28,28,1).astype('float32')

                re_x_test = _x_test.values.reshape(_x_test.shape[0], 28,28,1).astype('float32')

            else:

                re_x_train = _x_train.reshape(_x_train.shape[0], 28,28,1).astype('float32')

                re_x_test = _x_test.reshape(_x_test.shape[0], 28,28,1).astype('float32')



            re_y_train = np_utils.to_categorical(y_train, num_class)

            re_y_test = np_utils.to_categorical(y_test, num_class)

            

            try:

                model = load_model(model_path + 'cnn_model' + '_' + method + '.h5')

                print('Model Reading Success')

            except:

                ##model building

                model = Sequential()

                #convolutional layer with rectified linear unit activation

                model.add(Conv2D(32, kernel_size=(3, 3),   #32 convolution filters used each of size 3x3

                                 activation='relu',

                                 input_shape=(28,28,1)))

                model.add(Conv2D(64, (3, 3), activation='relu')) #64 convolution filters used each of size 3x3

                model.add(MaxPooling2D(pool_size=(2, 2))) #choose the best features via pooling

                #randomly turn neurons on and off to improve convergence

                model.add(Dropout(0.25))

                #flatten since too many dimensions, we only want a classification output

                model.add(Flatten())

                #fully connected to get all relevant data

                model.add(Dense(128, activation='relu'))

                #one more dropout for convergence' sake :) 

                model.add(Dropout(0.5))

                #output a softmax to squash the matrix into output probabilities

                model.add(Dense(num_class, activation='softmax'))

                #Model Compile

                model.compile(loss=keras.losses.categorical_crossentropy,

                              optimizer=keras.optimizers.Adadelta(),

                              metrics=['accuracy'])

                #Model Fitting

                history = model.fit(re_x_train,

                          re_y_train,

                          batch_size = 128,

                          epochs = 12,

                          verbose = 1,

                          validation_data = (re_x_test,

                                             re_y_test)

                         )

                #Model Saving

                #model.save(model_path + 'cnn_model' + '_' + method + '.h5')  

                hist_df = pd.DataFrame(history.history) # convert the history.history dict to a pandas DataFrame

                # save to json:  

                hist_json_file = history_path + 'cnn_model' + '_' + method + '_history.json' 

                with open(hist_json_file, mode='w') as f:

                    hist_df.to_json(f)

                

                

        else:

            raise NameError('Model Name Error')

            

        ###Fitting into the test data to get the score###---------------------

        print('Model Saved')

        print('Fitting the score')

        if model_name.upper() in ['SVM', 'LOGISTIC', 'DECISION TREE', 'RANDOM FOREST']:

            score = model.score(_x_test, y_test)

            print(model_name.upper() + '-' + method + "_score: "+ str(score))

            score_list[model_name + '_' + method] = score

            model_algo[method] = model

        if model_name.upper() in ['CNN']:

            score = model.evaluate(re_x_test, re_y_test, verbose = 0)

            print('Test loss:', score[0])

            print('Test accuracy:', score[1])

            score_list[model_name + '_' + method] = score[1]

            model_algo[method] = model

            

    ###To get the score DF -----------------------------------------------------

    score_df = pd.DataFrame.from_dict(score_list, orient='index').reset_index(drop = False)

    split = score_df["index"].str.split("_", n = 1, expand = True)

    score_df['model'] = split[0]

    score_df['scaling_method'] = split[1]

    score_df = score_df.drop(columns = ['index'])

    score_df.columns = ['score', 'model', 'scaling_method']

    score_df = score_df[['model', 'scaling_method', 'score']]

    score_df.to_excel(score_save_path +model_name + '_score.xlsx', index = False)

    print('Score is saved as Excel')

    return(score_df, model_algo)

    
svm_score, svm_algo = model_score('SVM', x_train, x_test, y_train, y_test)
log_score, log_algo = model_score('Logistic', x_train, x_test, y_train, y_test)
tree_score, tree_algo = model_score('Decision Tree', x_train, x_test, y_train, y_test)
rf_score, rf_algo = model_score('Random Forest', x_train, x_test, y_train, y_test)
cnn_score, cnn_algo = model_score('CNN', x_train, x_test, y_train, y_test)
#The below function is used to get back the CNN history training record,

#and even for plotting the training and testing trend

import json,codecs

def read_cnn_hist():

    data_scaling = ['original', 'Standarisation', 'Mean Normalisation', 'Unit']

    dict_hist = {}

    backup_path = '../input/digit-recognition-model-backup/'

    for scaling in data_scaling:

        try:

            path = backup_path + 'cnn_model_' + scaling + '_history.json'

            with codecs.open(path, 'r', encoding='utf-8') as f:

                    n = json.loads(f.read())

            dict_hist[scaling] = n

        except:

            print('No such ' + scaling + ' history file')

    return(dict_hist)

cnn_hist = read_cnn_hist()

cnn_hist
def show_train_history(history_dict):

    plt.subplots_adjust(wspace =0, hspace =0.7)

    plt.subplot(2,1,1)

    plt.title('Loss -- Train vs Test')

    plt.plot(history_dict['loss'].values())

    plt.plot(history_dict['val_loss'].values())

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['train', 'test'], loc='upper right')

    

    plt.subplot(2,1,2)

    plt.title('Acc -- Train vs Test')

    plt.plot(history_dict['acc'].values())

    plt.plot(history_dict['val_acc'].values())

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['train', 'test'], loc='upper right')



#Here I Just show the Unit Scaling data in training CNN history 

show_train_history(history_dict=cnn_hist['Unit'])
# Read all score excel

import glob

all_data = pd.DataFrame()

input_path = '../input/digit-recognition-model-backup/'

for f in glob.glob(input_path + "*.xlsx"):

    df = pd.read_excel(f)

    all_data = all_data.append(df,ignore_index=True)

all_data
# Show the comparison in table:

pivot_table = pd.pivot_table(all_data, index = ['scaling_method'], columns = ['model'], values = 'score').sort_index(ascending = False)

pivot_table.style.highlight_max()
#Hyperparameter Grid Search

#from sklearn.model_selection import GridSearchCV

#from sklearn.pipeline import make_pipeline

#pipe_svc = make_pipeline(svm.SVC(random_state = 1))

#param_range = [0.0001, 0.001, 0.1, 1.0, 100.0, 1000.0]

#param_grid = [{'svc__C' : param_range, 'svc__kernel' :['linear']},

#            {'svc__C' : param_range, 'svc__gamma' : param_range, 'svc__kernel' :['linear']}]

#gs = GridSearchCV(estimator = pipe_svc, param_grid= param_grid, scoring = 'accuracy', cv = 10 , n_jobs = 3)

#gs = gs.fit(X_train_std, x_label)
#Hyperparameter Grid Search

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score



pipe_tree = DecisionTreeClassifier(random_state = 1)

param_grid = {'max_depth':np.arange(3, 10)}

gs = GridSearchCV(estimator  = pipe_tree, param_grid = param_grid)

gs.fit(x_train, y_train)
from sklearn.model_selection import cross_val_score

cross_val_score(DecisionTreeClassifier(random_state = 1), x_train, y_train, scoring='accuracy', cv = 5)
unit_x_test = x_test.apply(lambda x: (x - 0)/(255 - 0))

unit_x_test = unit_x_test.values.reshape(unit_x_test.shape[0], 28,28,1).astype('float32')

prediction = cnn_algo['Unit']

prediction = prediction.predict_classes(unit_x_test)
prediction
#Plot y-train Data

plt.figure(num='cnn_test_fig',figsize=(10,10)) 

for row in range(1,26):

    plt.subplot(5, 5, row) #must be >0

    plt.title('Predicted Digit:' + str(prediction[row-1]))

    plt.axis('off')

    plt.imshow(x_test.reset_index(drop = True).loc[row-1].values.reshape(28,28))
# Predict the Test.csv for submission

# I would like to use "Unit" Scaling

x_submission = test_data.apply(lambda x: (x - 0)/(255 - 0)).values.reshape(test_data.shape[0], 28,28,1).astype('float32')

x_submission
prediction = cnn_algo['Unit']

predicted = prediction.predict_classes(x_submission)
predicted
#Plot Submission Test Data set

plt.figure(num='cnn_submission_fig',figsize=(10,10)) 

for row in range(1,26):

    plt.subplot(5, 5, row) #must be >0

    plt.title('Predicted Digit:' + str(predicted[row-1]))

    plt.axis('off')

    plt.imshow(test_data.loc[row-1].values.reshape(28,28))
#Submission

submission = pd.DataFrame.from_dict(dict(enumerate(predicted)), orient = 'index')

submission = submission.reset_index()

submission.columns = ['ImageId', 'Label']

submission['ImageId'] = submission['ImageId'] + 1

submission.to_csv('submission.csv', index = False)

submission