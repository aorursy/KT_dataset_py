# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import pickle
train_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv',

                      index_col='session_id')

test_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv',

                     index_col='session_id')
train_df.head(20)
train_df.info()
# Меняем тип атрибутов site1, ..., site10 на целочисленный и заменяем отсутствующие значения нулями

sites = ['site%s' % i for i in range(1,11)]

train_df[sites] = train_df[sites].fillna(0).astype(int)

test_df[sites] = test_df[sites].fillna(0).astype(int)
train_df.head()
with open(r"/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl", "rb") as input_file:

    site_dict = pickle.load(input_file)



sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
sites_dict.head()
sites_dict.shape
train_df.shape, test_df.shape
train_df['target'].values
y_train = train_df['target'].values
train_df[sites].to_csv('train_sessions_text.txt', 

                                 sep=' ', index=None, header=None)

test_df[sites].to_csv('test_sessions_text.txt', 

                                sep=' ', index=None, header=None)
!head train_sessions_text.txt
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(ngram_range=(1, 1), max_features=50000)

with open('train_sessions_text.txt') as inp_train_file:

    X_train = cv.fit_transform(inp_train_file)

with open('test_sessions_text.txt') as inp_test_file:

    X_test = cv.transform(inp_test_file)

print(X_train.shape, X_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
X_train_log, X_valid_log, y_train_log, y_valid_log = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
log_reg = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', max_iter=500)

log_reg.fit(X_train_log, y_train_log)
y_pred = log_reg.predict_proba(X_valid_log)

score_log = roc_auc_score(y_valid_log, y_pred[:,1])

print("log",score_log)
log_reg.fit(X_train, y_train)
# Делаем предсказания

y_test = log_reg.predict_proba(X_test)
y_test[:5]
def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
write_to_submission_file(y_predicted[:,1], 'baseline_1.csv')
####################### SGDClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier
########## Ансамбль на логистической регрессии, случайных деревьях и SGD классификаторе
random_forest_clf=RandomForestClassifier(n_estimators = 500,max_depth = 20 ,random_state = 42)

log_reg_clf=LogisticRegression(C=2.5, random_state=42, solver='saga', max_iter=500)

sgd_clf=SGDClassifier(alpha = 0.0005, eta0 = 0.01, learning_rate='adaptive', loss='modified_huber',

                      penalty ='l2',random_state = 42)

random_forest_clf.fit(X_train_log,y_train_log)

log_reg_clf.fit(X_train_log,y_train_log)

sgd_clf.fit(X_train_log,y_train_log)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(

                estimators=[('rf',random_forest_clf),('lr',log_reg_clf),('SVC',sgd_clf)],

                voting = 'soft')
voting_clf.fit(X_train_log,y_train_log)
y_predicted = voting_clf.predict_proba(X_valid_log)

score_vote = roc_auc_score(y_valid_log, y_predicted[:,1])

print("voting",score_vote)
voting_clf.estimators_
import tensorflow as tf

from tensorflow import keras
random_forest_clf=RandomForestClassifier(n_estimators = 500,max_depth = 20 ,random_state = 42)

log_reg_clf=LogisticRegression(C=2.5, random_state=42, solver='saga', max_iter=500)

sgd_clf=SGDClassifier(alpha = 0.0005, eta0 = 0.01, learning_rate='adaptive', loss='modified_huber',

                      penalty ='l2',random_state = 42)

random_forest_clf.fit(X_train_log,y_train_log)

log_reg_clf.fit(X_train_log,y_train_log)

sgd_clf.fit(X_train_log,y_train_log)
# Выбираем из обучающего набора только посещаемые сайты

my_X_train = train_df.loc[0:,['site1','site2','site3','site4','site5','site6','site7','site8','site9','site10']]#,'target']]
forest_pred=random_forest_clf.predict(X_train_log)

log_reg_clf_pred=log_reg_clf.predict(X_train_log)

SGD_pred=sgd_clf.predict(X_train_log)

final_pred_f=random_forest_clf.predict(X_test)

final_pred_l=log_reg_clf.predict(X_test)

final_pred_s=sgd_clf.predict(X_test)
print(forest_pred.shape,log_reg_clf_pred.shape,SGD_pred.shape,my_X_train.shape)

predict_only = DataFrame({"forest_pred":forest_pred[:]})

Forest_data = DataFrame({"forest_pred":forest_pred[:]})

Log_data = DataFrame({"Log_pred":log_reg_clf_pred[:]})

SGD_data = DataFrame({"SGD_pred":SGD_pred[:]})

logdata=DataFrame({"Log_pred":final_pred_l[:]})

sgddata=DataFrame({"SGD_pred":final_pred_s[:]})

predict_only = predict_only.join(Log_data)

predict_only = predict_only.join(SGD_data)

predict_final = DataFrame({"forest_pred":final_pred_f[:]})

predict_final = predict_final.join(logdata)

predict_final = predict_final.join(sgddata)

SGD_data.shape
X_train_cnn = my_X_train.join(Forest_data)

X_train_cnn = X_train_cnn.join(Log_data)

X_train_cnn = X_train_cnn.join(SGD_data)
model = keras.models.Sequential()

model.add(keras.layers.Input(3))

model.add(keras.layers.Dense(50,activation='relu'))

model.add(keras.layers.Dense(10,activation='relu'))

model.add(keras.layers.Dense(2,activation='softmax'))
from sklearn import metrics

from keras import backend as K

import tensorflow as tf

from sklearn.metrics import roc_auc_score



def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)





def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc

model.compile(loss="binary_crossentropy",optimizer="adam",

             metrics=["accuracy",auroc])
model.fit(predict_only,y_train_log,epochs=10,validation_split=0.1)
#y_predicted = model.predict_proba(X_train_cnn)

#score_vote = roc_auc_score(y_valid_cnn, y_predicted[:,1])

#print("voting",score_vote)

y_predicted=model.predict_proba(predict_final)
test_forest_pred=random_forest_clf.predict(X_valid_log)

test_log_reg_clf_pred=log_reg_clf.predict(X_valid_log)

test_SGD_pred=sgd_clf.predict(X_valid_log)
test_Forest_data = DataFrame({"forest_pred":test_forest_pred[:]})

test_Log_data = DataFrame({"Log_pred":test_log_reg_clf_pred[:]})

test_SGD_data = DataFrame({"SGD_pred":test_SGD_pred[:]})
X_valid_log.data