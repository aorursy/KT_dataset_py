import numpy as np 

import pandas as pd 

# Data processing, metrics and modeling

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf

import keras

from keras.wrappers.scikit_learn import KerasClassifier

from keras import models

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler,EarlyStopping

# Reproductibility

from numpy.random import seed

seed(84)

import gc

# tf.random.set_seed(84)

import os

os.listdir('../input/titanic')
####################################

# Importing data and merging

####################################



# Reading dataset

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



# Adding a column in each dataset before merging

train['Type'] = 'train'

test['Type'] = 'test'



# Merging train and test

data = train.append(test)



####################################

# Missing values and new features

####################################



# Title

data['Title'] = data['Name']



# Cleaning name and extracting Title

for name_string in data['Name']:

    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    

# Replacing rare titles 

mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 

           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 

           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 

           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}

           

data.replace({'Title': mapping}, inplace=True)

titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']



# Replacing missing age by median/title 

for title in titles:

    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]

    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute

    

# New feature : Family_size

data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

data.loc[:,'FsizeD'] = 'Alone'

data.loc[(data['Family_Size'] > 1),'FsizeD'] = 'Small'

data.loc[(data['Family_Size'] > 4),'FsizeD'] = 'Big'



# Replacing missing Fare by median/Pclass 

fa = data[data["Pclass"] == 3]

data['Fare'].fillna(fa['Fare'].median(), inplace = True)



#  New feature : Child

data.loc[:,'Child'] = 1

data.loc[(data['Age'] >= 18),'Child'] =0



# New feature : Family Survival (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5



data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

                               

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin == 0.0):

                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

                

for _, grp_df in data.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin == 0.0):

                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

                    

####################################

# Encoding and pre-modeling

####################################                  



# dropping useless features

data = data.drop(columns = ['Age','Cabin','Embarked','Name','Last_Name',

                            'Parch', 'SibSp','Ticket', 'Family_Size'])



# Encoding features

target_col = ["Survived"]

id_dataset = ["Type"]

cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()

cat_cols   = [x for x in cat_cols ]

# numerical columns

num_cols   = [x for x in data.columns if x not in cat_cols + target_col + id_dataset]

# Binary columns with 2 values

bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

# Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    data[i] = le.fit_transform(data[i])

# Duplicating columns for multi value columns

data = pd.get_dummies(data = data,columns = multi_cols )

# Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(data[num_cols])

scaled = pd.DataFrame(scaled,columns = num_cols)

# dropping original values merging scaled values for numerical columns

df_data_og = data.copy()

data = data.drop(columns = num_cols,axis = 1)

data = data.merge(scaled,left_index = True,right_index = True,how = "left")

data = data.drop(columns = ['PassengerId'],axis = 1)



# Target = 1st column

cols = data.columns.tolist()

cols.insert(0, cols.pop(cols.index('Survived')))

data = data.reindex(columns= cols)



# Cutting train and test

train = data[data['Type'] == 1].drop(columns = ['Type'])

test = data[data['Type'] == 0].drop(columns = ['Type'])



# X and Y

X_train = train.iloc[:, 1:20].as_matrix()

y_train = train.iloc[:,0].as_matrix()
train.head()
def make_model():

    model = models.Sequential()

#     model.add(Dense(16, activation='relu',input_shape=(X_train.shape[1],)))

#     model.add(Dropout(0.3))

#     model.add(BatchNormalization())

#     model.add(Dense(8,activation='relu'))

#     model.add(Dropout(0.2))

#     model.add(BatchNormalization())

#     model.add(Dense(4,activation='relu'))

#     model.add(Dropout(0.2))

#     model.add(BatchNormalization())

#     model.add(Dense(1,activation='sigmoid'))

    model.add(Dense(13, input_dim = 18, activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Dense(8, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model

def mixup_data(x, y, alpha=1.0):

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1



    sample_size = x.shape[0]

    index_array = np.arange(sample_size)

    np.random.shuffle(index_array)

    

    mixed_x = lam * x + (1 - lam) * x[index_array]

    mixed_y = (lam * y) + ((1 - lam) * y[index_array])



    return mixed_x, mixed_y



def make_batches(size, batch_size):

    nb_batch = int(np.ceil(size/float(batch_size)))

    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]





def batch_generator(X,y,batch_size=128,shuffle=True,mixup=False):

    sample_size = X.shape[0]

    index_array = np.arange(sample_size)

    

    while 1:

        if shuffle:

            np.random.shuffle(index_array)

        batches = make_batches(sample_size, batch_size)

        for batch_index, (batch_start, batch_end) in enumerate(batches):

            batch_ids = index_array[batch_start:batch_end]

            X_batch = X[batch_ids]

            y_batch = y[batch_ids]

            

            if mixup:

                X_batch,y_batch = mixup_data(X_batch,y_batch,alpha=1.0)

                

            yield X_batch,y_batch
def step_decay(epoch):

   initial_lrate = 0.01

   drop = 0.5

   epochs_drop = 10.0

#    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

   lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))

   return lrate

lrate = LearningRateScheduler(step_decay)
class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

       self.losses = []

       self.lr = []

 

    def on_epoch_end(self, batch, logs={}):

       self.losses.append(logs.get('loss'))

       self.lr.append(step_decay(len(self.losses)))
def auc(y_true, y_pred):

    try:

        return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

    except:

        return 0.5
batch_size = 32

loss_history = LossHistory()

lrate = LearningRateScheduler(step_decay)

annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

callbacks_list = [EarlyStopping(monitor='val_loss', patience=10,mode='min'),loss_history, annealer]

sss = StratifiedShuffleSplit(n_splits=5)

hold_models = []

hold_models_no_mixup = []

for train_index, test_index in sss.split(X_train, y_train):

    X_train_hold, X_val_hold = X_train[train_index], X_train[test_index]

    y_train_hold, y_val_hold = y_train[train_index], y_train[test_index]

    

    ### train with mixup ###

    tr_gen = batch_generator(X_train_hold,y_train_hold,batch_size=batch_size,shuffle=True,mixup=True)

    model = make_model()

    model.fit_generator(

            tr_gen, 

            steps_per_epoch=np.ceil(float(len(X_train_hold)) / float(batch_size)),

            nb_epoch=1000, 

            verbose=1, 

            callbacks=callbacks_list, 

            validation_data=(X_val_hold,y_val_hold),

            max_q_size=10,

            )

    hold_models.append(model)

    

    ### train without mixup ###

    tr_gen_no_mixup = batch_generator(X_train_hold,y_train_hold,batch_size=batch_size,shuffle=True,mixup=False)

    model_no_mixup = make_model()

    model_no_mixup.fit_generator(

            tr_gen, 

            steps_per_epoch=np.ceil(float(len(X_train_hold)) / float(batch_size)),

            nb_epoch=1000, 

            verbose=1, 

            callbacks=callbacks_list, 

            validation_data=(X_val_hold,y_val_hold),

            max_q_size=10,

            )

    hold_models_no_mixup.append(model_no_mixup)

    

    del X_train_hold, X_val_hold, y_train_hold, y_val_hold

    gc.collect()
import matplotlib.pyplot as plt

%matplotlib inline



def append_last_value(target_list, length):

    if len(target_list) >= length:

        return target_list

    for i in range(length - len(target_list)):

        target_list.append(target_list[-1])

    return target_list





model_index = 1

trained_epochs = max(len(hold_models[model_index].history.history['loss']), len(hold_models_no_mixup[model_index].history.history['loss']))



loss_list = append_last_value(hold_models[model_index].history.history['loss'], trained_epochs)

val_acc_list = append_last_value(hold_models[model_index].history.history['val_accuracy'], trained_epochs)



loss_list_no_mixup = append_last_value(hold_models_no_mixup[model_index].history.history['loss'], trained_epochs)

val_acc_list_no_mixup = append_last_value(hold_models_no_mixup[model_index].history.history['val_accuracy'], trained_epochs)



plt.plot(range(1, trained_epochs+1), loss_list, label="loss")

plt.plot(range(1, trained_epochs+1), val_acc_list, label="val_acc")

plt.plot(range(1, trained_epochs+1), loss_list_no_mixup, label="loss_no_mixup")

plt.plot(range(1, trained_epochs+1), val_acc_list_no_mixup, label="val_accuracy_no_mixup")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
def get_pred(model_list, test_df):

    preds = [model.predict(test_df.drop('Survived', axis=1)) for model in model_list]

    model_count = len(preds)

    ensemble_pred = 0

    for i in range(model_count):

        ensemble_pred = ensemble_pred + preds[i]

    ensemble_pred = ensemble_pred / model_count

    pred_ = ensemble_pred[:,0]

    return pred_



pred = get_pred(hold_models, test)

pred_no_mixup = get_pred(hold_models_no_mixup, test)
# fig = plt.figure()

# ax = fig.add_subplot(1,1,1)



# tmp_pred = hold_models[model_index].predict(test.drop('Survived', axis=1))

# tmp_pred_no_mixup = hold_models_no_mixup[model_index].predict(test.drop('Survived', axis=1))



# # ax.hist(tmp_pred, bins=10, density=True, color='red', alpha=0.5)

# # ax.hist(tmp_pred_no_mixup, bins=10, density=True, color='blue',alpha=0.5)

# ax.hist(tmp_pred, bins=10, color='red', alpha=0.5)

# ax.hist(tmp_pred_no_mixup, bins=10, color='blue',alpha=0.5)

# ax.set_title('histogram of pred score')

# ax.set_xlabel('pred')

# ax.set_ylabel('count')

# ax.set_ylim(0,0.1)

# fig.show()
train_survived = train['Survived'].sum()

train_len = train.shape[0]

print('train dataset : ' + str(train_survived) + '/' + str(train_len) + ' : ' + str(train_survived / train_len))



thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8]

for threshold in thresholds:

    pred_survived = pred[pred > threshold].shape[0]

    pred_len = pred.shape[0]

    print('threshold : ' + str(threshold) + ' -> ' + str(pred_survived) + '/' + str(pred_len) + ' : ' + str(pred_survived / pred_len))
submit_threshold = 0.4

submit_df = pd.read_csv('../input/titanic/test.csv')

submit_df['Survived'] = np.where(pred > submit_threshold, 1, 0)

submit_df = submit_df[['PassengerId', 'Survived']]

submit_df.to_csv('submission.csv', index=False)