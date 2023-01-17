import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# Loading data

df = pd.read_csv('../input/creditcard.csv')



print("Number of samples: {}".format(len(df)))

print("Number of attributes: {}".format(len(df.columns)))
df.describe()
df.sample(5)
df.groupby("Class").count()["Time"]
y = df['Class'].copy()

X = df.copy()

del X['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



c_param = 0.1 #should use cross valid to find this

lr = LogisticRegression(C = c_param, penalty = 'l2', class_weight ='balanced', max_iter =100)

lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)

mat_conf = confusion_matrix(y_test, y_pred)

mat_conf
print("We have {0} well detected fraud (True positives)".format(mat_conf[1,1]))

print("We have {0} undetected fraud (False negatives)".format(mat_conf[1,0]))

print("We have {0} normal behavior classified as fraud (Flase positives)".format(mat_conf[0,1]))

lr_weight = LogisticRegression(C = c_param, penalty = 'l2', class_weight ={0:1,1:400}, max_iter =100)

lr_weight.fit(X_train, y_train)



y_pred_weight = lr_weight.predict(X_test)

mat_conf_weight = confusion_matrix(y_test, y_pred_weight)

mat_conf_weight
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_test, label=y_test)

#param = {'num_leaves': 31, 'objective': 'binary', "verbosity" : 1}

#param = {'objective': 'binary', "verbosity" : 1}



param = {'objective': 'binary', 

         "verbosity" : 1,

         #"is_unbalance" : True,

         #"max_bin" : 40,

         'learning_rate' : 0.001,

        } #even using this params yields the same results



param = {'objective': 'binary', "verbosity" : 1,"learning_rate" : 0.01 }





param['metric'] = ['auc', 'binary_logloss','cross_entropy']

evals_result={}
nb_rounds = 1000

verbose_eval = int(nb_rounds/10)

gbm = lgb.train(param,

                train_data,

                num_boost_round=nb_rounds,

                valid_sets=[train_data, test_data],

                evals_result=evals_result,

                verbose_eval=verbose_eval,

                #early_stopping_rounds=50

               )
ax = lgb.plot_metric(evals_result, metric='auc')

ax = lgb.plot_metric(evals_result, metric='binary_logloss')

ax = lgb.plot_metric(evals_result, metric='xentropy')



plt.show()
fig, ax = plt.subplots(figsize=(12,10))

lgb.plot_importance(gbm, height=0.8, ax=ax)

ax.grid(False)

plt.ylabel('Feature', size=12)

plt.xlabel('Importance', size=12)

plt.title("Importance of the Features of our LightGBM Model", fontsize=15)

plt.show()
ypred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)

threshold = 0.75

mat_conf_lgb = confusion_matrix(y_test, ypred_lgb>threshold)

mat_conf_lgb
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from inspect import signature

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(y_test, ypred_lgb)



precision, recall, _ = precision_recall_curve(y_test, ypred_lgb)



# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(

          average_precision))
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, ypred_lgb)

roc_auc = auc(fpr, tpr)

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import SGD

from keras import backend as K

import matplotlib.pyplot as plt 

from sklearn.utils import shuffle



def show_train_history(train_history,train,validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='best')

    plt.show()



def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall
df.Class.value_counts()
df.sort_values(by='Class', ascending=False, inplace=True) #easier for stratified sampling

#df_full.drop('Time', axis=1,  inplace = True)

df_sample = df.iloc[:3000,:]

shuffle_df = shuffle(df_sample, random_state=42)

df_sample.Class.value_counts()

y = shuffle_df['Class'].copy()

X = shuffle_df.copy()

del X['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



scaler.fit(x_train)

x_train_trans = scaler.transform(x_train)

x_test_trans = scaler.transform(x_test)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



model = Sequential()

model.add(Dense(200, input_dim=30, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(200, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



sgd_opti = SGD(lr=10, momentum=0.01, decay=0.0, nesterov=False)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                   patience=5, verbose=1, mode='min',

                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)



early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,

                      patience=30) # probably needs to be more patient, but kaggle time is limited



callbacks_list = [early, reduceLROnPlat]





model.compile( loss ="binary_crossentropy",  #loss='binary_crossentropy',

              optimizer = 'adam', #optimizer = sgd_opti, #optimizer='rmsprop',

              metrics=['accuracy', precision, recall])



ratio = df.groupby("Class").count()["Time"][0] / df.groupby("Class").count()["Time"][1]



class_weight = {0: 5,

                1: 1}



train_history = model.fit(x_train_trans,y_train,

          epochs=200,

          validation_split=0.2,

          batch_size=512,

          class_weight=class_weight)
show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')

show_train_history(train_history,'precision','val_precision')

show_train_history(train_history,'recall','val_recall')



score = model.evaluate(x_test_trans, y_test, batch_size=128)
print(score)
y_pred = model.predict_classes(x_test_trans)

mat_conf = confusion_matrix(y_test, y_pred.astype(int))

mat_conf
y = df['Class'].copy()

X = df.copy()

del X['Class']

X_TEST = scaler.transform(X)



score = model.evaluate(X_TEST, y, batch_size=128)

print(score)

Y_PRED = model.predict_classes(X_TEST)

mat_conf_all_dataset = confusion_matrix(y,Y_PRED.astype(int))

mat_conf_all_dataset

def custom_loss(y_true,y_pred):

    bce = K.mean(K.binary_crossentropy(y_true,y_pred),axis=-1)

    return bce
model.compile( loss ="binary_crossentropy",  #loss='binary_crossentropy',

              optimizer = 'adam', #optimizer = sgd_opti, #optimizer='rmsprop',

              metrics=['accuracy', precision, recall])





train_history = model.fit(x_train_trans,y_train,

          epochs=200,

          validation_split=0.5,

          batch_size=512,

          class_weight=class_weight,

        callbacks=callbacks_list,)
show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')

show_train_history(train_history,'precision','val_precision')

show_train_history(train_history,'recall','val_recall')



score = model.evaluate(x_test_trans, y_test, batch_size=128)
y_pred = model.predict_classes(x_test_trans)

mat_conf = confusion_matrix(y_test, y_pred.astype(int))

mat_conf
y = df['Class'].copy()

X = df.copy()

del X['Class']

X_TEST = scaler.transform(X)



score = model.evaluate(X_TEST, y, batch_size=128)

print(score)

Y_PRED = model.predict_classes(X_TEST)

mat_conf_all_dataset = confusion_matrix(y,Y_PRED.astype(int))

mat_conf_all_dataset