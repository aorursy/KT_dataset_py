#making the imports

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

#reading the csv file provided

data = pd.read_csv('../input/creditcard.csv')
#checking the head

data.head()
#having a look at the column data types 

data.info()
#checking for nulls.



data.isnull().sum()
#checking the target variable distribution

plt.figure(figsize = (8,6))

sns.set_style('dark')

sns.countplot(data['Class'])

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(data['Class'])

plt.yscale('log')

plt.show()
#dataset histogram

data.hist(figsize = (20,20))

plt.show()
#checking the distribution of transaction amount and time



plt.figure(figsize = (16,6))

plt.subplot(1,2,1)

sns.distplot(data['Amount'], color='r', kde= False)

plt.title('Distribution of Transaction Amount', fontsize=16)

plt.yscale('log')



plt.subplot(1,2,2)

sns.distplot(data['Time'], color='b', kde= False)

plt.title('Distribution of Transaction Time', fontsize=16)

plt.yscale('log')



plt.show()

#checking the percentage of fraud transactions



fraud = data[data['Class'] == 1]

valid = data[data['Class'] == 0]



fraud_ratio = len(fraud)/ float(len(valid) + len(fraud))



print('The number of Fraudulent cases is: {} \n'.format(len(fraud)))

print('The number of valid transactions is: {} \n'.format(len(valid)))

print('The ratio of fraudulent transactions is: {}'.format(fraud_ratio))
#heat map of data to see correlation



plt.figure(figsize = (14,10))

sns.heatmap(data.corr(), cmap= 'coolwarm')

plt.show()
#lets divide the data into X and y



columns = data.columns.tolist()



cols = [c for c in columns if c not in ['Class']]

target = 'Class'



X = data[cols]

y = data[target]



print(X.shape)

print(y.shape)
#scale the data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaled_X = scaler.fit_transform(X)
#convert target variable to numpy array

y = y.as_matrix()
#print the shapes of X and y

print(scaled_X.shape)

print('\n')

print(y.shape)
#make the imports

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
#doing the train test split (20% test data)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.20, random_state=101)
#fit the data to default Ramdom Forest classifier.

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
#get the predictions

pred_Random_Forest = rfc.predict(X_test)
#function for plotting confusion matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.coolwarm):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize = 20)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0, fontsize = 20)

    plt.yticks(tick_marks, classes, fontsize = 20)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1#print('Confusion matrix, without normalization')

        #print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black",

                 fontsize=25)



    plt.tight_layout()

    plt.ylabel('True label', fontsize = 20)

    plt.xlabel('Predicted label', fontsize = 20)
#confusion matrix

cnf_matrix = confusion_matrix(y_test,pred_Random_Forest)

np.set_printoptions(precision=2)
#print the precision, recall , accuracy and confusion matrix



from sklearn.metrics import precision_score, recall_score, accuracy_score



print('Precision Score: {}\n'.format(precision_score(y_test,pred_Random_Forest)))

print('Recall Score: {}\n'.format(recall_score(y_test,pred_Random_Forest)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_Random_Forest)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()
# import and instantiate the grid search cv

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold



fold = KFold(n_splits= 3, shuffle= True, random_state= 42)



param_grid = {

    'max_depth': [8,10,12],

    'min_samples_leaf': range(100, 200, 300),

    'min_samples_split': range(200, 300, 400),

    'n_estimators': [50,100,200]

}

# Create a base model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = fold, n_jobs = -1,verbose = 1)
#fit the grid (will take some time)

grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
#lets predict with these parameters



random_final = RandomForestClassifier(bootstrap= True, 

                                      max_depth=12,

                                      min_samples_leaf=100, 

                                      min_samples_split=200,

                                      n_estimators=50)



random_final.fit(X_train,y_train)
#make predictions with this model

pred_3 = random_final.predict(X_test)
#confusion matrix

conf_matt = confusion_matrix(y_test,pred_3)
#print the precision, recall , accuracy and confusion matrix



print('Precision Score: {}\n'.format(precision_score(y_test,pred_3)))

print('Recall Score: {}\n'.format(recall_score(y_test,pred_3)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_3)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(conf_matt

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()
#lets use logistic regression model



from sklearn.linear_model import LogisticRegression



#after multiple runs below parameters seem to give the best restult. 



log_reg = LogisticRegression(C = 0.01, penalty= 'l2')



log_reg.fit(X_train,y_train)



pred_log_reg = log_reg.predict(X_test)



conf_mat = confusion_matrix(y_test,pred_log_reg)





#print the precision, recall , accuracy and confusion matrix



print('Precision Score: {}\n'.format(precision_score(y_test,pred_log_reg)))

print('Recall Score: {}\n'.format(recall_score(y_test,pred_log_reg)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_test,pred_log_reg)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(conf_mat

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()
#making the imports

import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance
#try xgboost with default parameters



xgb_model = XGBClassifier()



xgb_model.fit(X_train,y_train)



xgb_pred = xgb_model.predict(X_test)



conf_mat = confusion_matrix(y_test,xgb_pred)





#print the precision, recall , accuracy and confusion matrix



print('Precision Score: {}\n'.format(precision_score(y_test,xgb_pred)))

print('Recall Score: {}\n'.format(recall_score(y_test,xgb_pred)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_test,xgb_pred)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(conf_mat

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()

# hyperparameter tuning with XGBoost (will take some time to run)



# creating a KFold object 

folds = KFold(n_splits= 3, shuffle= True, random_state= 101)



# specify range of hyperparameters

param_grid = {'learning_rate': [0.1,0.2, 0.6], 

             'subsample': [0.3, 0.6, 0.9]}          





# specify model

xgb_model = XGBClassifier(max_depth=2, n_estimators=200)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'roc_auc', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True, 

                       n_jobs= -1)      

model_cv.fit(X_train,y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',model_cv.best_score_,'using',model_cv.best_params_)
#lets use model with these parameters

xgb_model_2 = XGBClassifier(max_depth=2, n_estimators=200, learning_rate= 0.1, subsample= 0.6)
#lets make predictions using this model

xgb_model_2.fit(X_train,y_train)



xgb_pred = xgb_model_2.predict(X_test)



conf_mat = confusion_matrix(y_test,xgb_pred)





#print the precision, recall , accuracy and confusion matrix



print('Precision Score: {}\n'.format(precision_score(y_test,xgb_pred)))

print('Recall Score: {}\n'.format(recall_score(y_test,xgb_pred)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_test,xgb_pred)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(conf_mat

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()
#making the imports



import tensorflow as tf

import itertools

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils.np_utils import to_categorical
# checking the shapes of test and train

print(np.shape(X_train))

print(np.shape(y_train))

print(np.shape(X_test))

print(np.shape(y_test))
#define the model

model = Sequential()

model.add(Dense(64, input_dim=30, activation='relu'))

model.add(Dropout(0.9))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.9))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.9))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.9))

model.add(Dense(2, activation='softmax'))  # With 2 outputs

#compile using adam optimizer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#convert y_train and y_test to categorical values with 2 classes

y_train = to_categorical(y_train, num_classes = 2)

y_test = to_categorical(y_test, num_classes = 2)
#lets do first 10 epochs with a batch size of 2048

epoch = 10

batch_size = 2048

model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size)
#evaluate the model

score, acc = model.evaluate(X_test, y_test)

print('Test score:', score)

print('Test accuracy:', acc)
#training for more epochs to get better results (lets say 50 epochs)

history = model.fit(X_train, y_train, batch_size = 2048, epochs = 50, 

         validation_data = (X_test, y_test), verbose = 2)
# Check the history keys

history.history.keys()
#convert to a data frame

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch



#make the epoch start from 1

def add_one(x):

    return x+1



hist['epoch'] = hist['epoch'].apply(add_one)



hist
#plotting the results to see difference between train and validation accuracy/loss



plt.figure(figsize = (14,6))

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.plot(hist['epoch'],hist['val_acc'], label = 'Val Accuracy')

plt.plot(hist['epoch'],hist['acc'], label = 'Train Accuracy')

plt.xticks(range(1,51))

plt.legend(loc = 'lower right')

plt.title('Accuracy')

plt.show()



plt.figure(figsize = (14,6))

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.plot(hist['epoch'],hist['val_loss'], label = 'Val Loss')

plt.plot(hist['epoch'],hist['loss'], label = 'Train Loss')

plt.xticks(range(1,51))

plt.legend()

plt.title('Loss')

plt.show()
#lets predict using this trained model and get the confusion matrix



nn_pred = model.predict(X_test)



pred_classes = np.argmax(nn_pred,axis = 1)



y_true = np.argmax(y_test,axis = 1) 



conf_mat = confusion_matrix(y_true,pred_classes)





#print the precision, recall , accuracy and confusion matrix



print('Precision Score: {}\n'.format(precision_score(y_true,pred_classes)))

print('Recall Score: {}\n'.format(recall_score(y_true,pred_classes)))

print('Accuracy Score: {}\n'.format(accuracy_score(y_true,pred_classes)))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure(figsize = (8,6))

plot_confusion_matrix(conf_mat

                      , classes=class_names

                      , title='CONF MATRIX')

plt.show()