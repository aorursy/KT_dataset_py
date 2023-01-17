import os

import numpy as np

import pandas as pd

from time import time

import random

import sklearn.utils as sf

import sklearn.model_selection as ms

import sklearn.metrics as eva

from sklearn.preprocessing import normalize, LabelEncoder

from sklearn.linear_model import SGDClassifier



from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers
def data_selection(way, obj):

    if way == 'random':

        return sf.shuffle(obj, n_samples = 1)[0]

    elif way == 'max':

        return np.max(obj)

    elif way == 'min':

        return np.min(obj)

    elif way == 'average':

        return np.average(obj)

    

def data_multiplication(time, profile):

    row_num = list(set([globals()[i].shape[0] for i in data_list]))[0]

    x_array = np.zeros(len(data_list))

    temp_profile = np.zeros(profile.shape[1])

    for k in range(time):

        temp_x_array = np.zeros(row_num)

        temp_x_array = temp_x_array.reshape(-1,1)

        for i in data_list:

            temp_array = np.array([data_selection('random', globals()[i][j]) for j in range(row_num)])

            temp_x_array = np.hstack((temp_x_array, temp_array.reshape(-1,1)))

        temp_x_array = temp_x_array[:,1:]

        x_array = np.vstack((x_array, temp_x_array))

        temp_profile = np.vstack((temp_profile, profile))

    return normalize(x_array[1:,:]), temp_profile[1:,].astype(str)



def target_data(target):

    if target == 'cooler':

        return profile[:,0]

    elif target == 'valve':

        return profile[:,1]

    elif target == 'pump':

        return profile[:,2]

    elif target == 'accumulator':

        return profile[:,3]

    elif target == 'stability':

        return profile[:,4]

    

def fit_pred_eva(way):

    if way == 'nn':

        

        for i in ['Y_train', 'Y_test']:

            encoder = LabelEncoder()

            encoder.fit(globals()[i])

            globals()[i+"_encoded"] = encoder.transform(globals()[i])

            globals()[i+"_categorized"] = np_utils.to_categorical(globals()[i+"_encoded"])

        

        model = Sequential()

        model.add(Dense(39, input_dim=17, activation='relu'))

        model.add(Dense(13, activation='relu'))

        model.add(Dense(8, activation='relu'))

        model.add(Dense(Y_train_categorized.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

        

        start_time = time()

        model.fit(X_train, Y_train_categorized, epochs=10, batch_size=10)

        _, pred_score = model.evaluate(X_test, Y_test_categorized)

        end_time = time()

        

        time_spend = np.round(end_time - start_time, 5)

        return np.round(pred_score * 100, 3), time_spend

    else:

        if way == 'xgboost':

            model = XGBClassifier()

        elif way == 'svm':

            model = SVC()

        elif way == 'randomforest':

            model = RandomForestClassifier()

        elif way == 'decisiontree':

            model = DecisionTreeClassifier()

        elif way == 'knn':

            model = KNeighborsClassifier()

        elif way == 'sgd':

            model = SGDClassifier()

        

        start_time = time()

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        pred_score = 100 * np.round(eva.accuracy_score(Y_test, Y_pred),5)

        end_time = time()

        

        time_spend = np.round(end_time - start_time, 3)

    return pred_score, time_spend
file_location = '/kaggle/input/condition-monitoring-of-hydraulic-systems/'

file_list = os.listdir(file_location)

data_list = []



for i in file_list:

    try:

        globals()[i[:-4]] = np.loadtxt(file_location+i, delimiter = '\t')

        data_list.append(i[:-4])

    except:

        pass

    

data_list.remove('profile')
x_array, profile = data_multiplication(1,profile)

X_train, X_test, Y_train, Y_test = ms.train_test_split(x_array, target_data('cooler'), test_size = 0.9, random_state = random.randint(0,1000))
models = ['sgd', 'randomforest', 'xgboost', 'svm', 'decisiontree', 'knn', 'nn']

result_table = pd.DataFrame(index = ['prediction accuracy (%)','time spend (s)'])



for i in models:

    pred_score, time_spend = fit_pred_eva(i)

    result_table[i] = [pred_score, time_spend]

    print(i+" completed")





result_table