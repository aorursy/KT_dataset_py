#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Apr  1 19:58:18 2020



@author: Jaehoon Cha 

@email: chajaehoon79@gmail.com

"""

import pandas as pd

import numpy as np

import os



'''

call data

'''

mother_given_path = "/kaggle/input/covid19-global-forecasting-week-2"



train_given_path = os.path.join(mother_given_path, "train.csv")

test_given_path = os.path.join(mother_given_path, "test.csv")



train = pd.read_csv(train_given_path)

test = pd.read_csv(test_given_path)





train.columns.values[2] = 'Country'

train.columns.values[1] = 'Subplace'

train.fillna("", inplace = True)



test.columns.values[2] = 'Country'

test.columns.values[1] = 'Subplace'

test.fillna("", inplace = True)





Countries = train.Country.unique()





'''

Set locations and population columns

'''

import pickle

mother_path = "/kaggle/input/covid19week2"

loc2pop_path = os.path.join(mother_path, "loc2popv2.pickle")

with open(loc2pop_path, 'rb') as f:

    loc2pop = pickle.load(f)





def make_data():   

    train.reset_index(drop = True, inplace = True)    

    

    

    train['Location'] = train[['Subplace', 'Country']].apply(lambda x: '_'.join(x), axis=1)

    for i in range(len(train)):

        if i%100 == 0:

            print(i/len(train))

        if train.loc[i, 'Subplace'] == "":

            train.loc[i, 'Location'] = train.loc[i, 'Country']

        elif train.loc[i, 'Subplace'] == train.loc[i, 'Country']:

            train.loc[i, 'Location'] = train.loc[i, 'Country']

        else:

            train.loc[i, 'Location'] = '_'.join([train.loc[i, 'Country'], train.loc[i, 'Subplace']])

        train.loc[i, 'Population'] = loc2pop[train.loc[i, 'Location']]

    

    

    test.reset_index(drop = True, inplace = True)    

    

    

    test['Location'] = test[['Subplace', 'Country']].apply(lambda x: '_'.join(x), axis=1)

    for i in range(len(test)):

        if i%100 == 0:

            print(i/len(test))

        if test.loc[i, 'Subplace'] == "":

            test.loc[i, 'Location'] = test.loc[i, 'Country']

        elif test.loc[i, 'Subplace'] == test.loc[i, 'Country']:

            test.loc[i, 'Location'] = test.loc[i, 'Country']

        else:

            test.loc[i, 'Location'] = '_'.join([test.loc[i, 'Country'], test.loc[i, 'Subplace']])

        test.loc[i, 'Population'] = loc2pop[test.loc[i, 'Location']]

    

    with open('train_dfv2.csv', 'wb') as f:

        pickle.dump(train, f)

    

    with open('test_dfv2.csv', 'wb') as f:

        pickle.dump(test, f)   





#make_data()





'''

Call dataset

'''

mother_path = "/kaggle/input/covid19week2"

traindf2_path = os.path.join(mother_path, "train_dfv2.csv")

testdf2_path = os.path.join(mother_path, "test_dfv2.csv")



with open(traindf2_path, 'rb') as f:

    train = pickle.load(f)



with open(testdf2_path, 'rb') as f:

    test = pickle.load(f)    

    

    

Locations = train.Location.unique()



'''

Check implausible data

'''

dic = {}

for loc in Locations:

    indexNames = train[train.Location == loc].index

    dic[loc] = np.array(train.loc[indexNames, "ConfirmedCases"])

    tmp = np.diff(dic[loc]) < 0

    if len(np.where(tmp == True)[0]) > 0:

        print(loc)    

    

dic = {}

for loc in Locations:

    indexNames = train[train.Location == loc].index

    dic[loc] = np.array(train.loc[indexNames, "Fatalities"])

    tmp = np.diff(dic[loc]) < 0

    if len(np.where(tmp == True)[0]) > 0:

        print(loc)



'''

Make dataset again by replacing implausible data

'''





def make_data():   

    train.reset_index(drop = True, inplace = True)    

    

    

    train['Location'] = train[['Subplace', 'Country']].apply(lambda x: '_'.join(x), axis=1)

    for i in range(len(train)):

        if i%100 == 0:

            print(i/len(train))

        if train.loc[i, 'Subplace'] == "":

            train.loc[i, 'Location'] = train.loc[i, 'Country']

        elif train.loc[i, 'Subplace'] == train.loc[i, 'Country']:

            train.loc[i, 'Location'] = train.loc[i, 'Country']

        else:

            train.loc[i, 'Location'] = '_'.join([train.loc[i, 'Country'], train.loc[i, 'Subplace']])

        train.loc[i, 'Population'] = loc2pop[train.loc[i, 'Location']]

    

    

    test.reset_index(drop = True, inplace = True)    

    

    

    test['Location'] = test[['Subplace', 'Country']].apply(lambda x: '_'.join(x), axis=1)

    for i in range(len(test)):

        if i%100 == 0:

            print(i/len(test))

        if test.loc[i, 'Subplace'] == "":

            test.loc[i, 'Location'] = test.loc[i, 'Country']

        elif test.loc[i, 'Subplace'] == test.loc[i, 'Country']:

            test.loc[i, 'Location'] = test.loc[i, 'Country']

        else:

            test.loc[i, 'Location'] = '_'.join([test.loc[i, 'Country'], test.loc[i, 'Subplace']])

        test.loc[i, 'Population'] = loc2pop[test.loc[i, 'Location']]

    

   

    for d in [6, 7, 8, 9]:

        train.loc[(train.Location == 'Australia_Northern Territory') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 1.0   

    for d in [31]:

        train.loc[(train.Location == 'Australia_Queensland') & (train.Date == '2020-01-{:02}'.format(d)), 'ConfirmedCases'] = 3.0   

    for d in [2, 3]:

        train.loc[(train.Location == 'Australia_Queensland') & (train.Date == '2020-02-{:02}'.format(d)), 'ConfirmedCases'] = 3.0  

    for d in [25]:

        train.loc[(train.Location == 'Canada_Alberta') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 433.0  

    for d in [17]:

        train.loc[(train.Location == 'China_Guizhou') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 146.0  

    for d in [9, 10, 11, 12, 13, 14, 15]:

        train.loc[(train.Location == 'France_Saint Barthelemy') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 3.0  

    for d in [17, 18, 19, 20, 21, 22, 23, 24, 25,26]:

        train.loc[(train.Location == 'Guyana') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 5.0  

    for d in [14]:

        train.loc[(train.Location == 'US_Alaska') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 1.0  

    for d in [18]:

        train.loc[(train.Location == 'US_Nevada') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 76.0  

    for d in [20]:

        train.loc[(train.Location == 'US_Utah') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 108.0  

    for d in [20, 21]:

        train.loc[(train.Location == 'US_Virgin Islands') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 3.0  

    for d in [29, 30]:

        train.loc[(train.Location == 'US_Virgin Islands') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 22.0  

    for d in [18]:

        train.loc[(train.Location == 'US_Washington') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 1226.0  

    for d in [21]:

        train.loc[(train.Location == 'Canada_Quebec') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 4.0  

    for d in [15]:

        train.loc[(train.Location == 'Iceland') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 0.0  

    for d in [20]:

        train.loc[(train.Location == 'Iceland') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 1.0  

    for d in [20]:

        train.loc[(train.Location == 'India') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 4.0  

    for d in [20]:

        train.loc[(train.Location == 'Kazakhstan') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 0.0  

    for d in [18]:

        train.loc[(train.Location == 'Philippines') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 14.0  

    for d in [18, 19, 20, 21]:

        train.loc[(train.Location == 'Slovakia') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 0.0  

    for d in [24]:

        train.loc[(train.Location == 'US_Hawaii') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 0.0  

    for d in [26, 27]:

        train.loc[(train.Location == 'Serbia') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 4.0  

    for d in [30]:

        train.loc[(train.Location == 'US_Virginia') & (train.Date == '2020-03-{:02}'.format(d)), 'Fatalities'] = 25.0  

       

    with open('train_dfv2.csv', 'wb') as f:

        pickle.dump(train, f)

    

    with open('test_dfv2.csv', 'wb') as f:

        pickle.dump(test, f)   





#make_data()







'''

Call data

'''



mother_path = "/kaggle/input/covid19week2"



train_path = os.path.join(mother_path, "train_dfv2.csv")

test_path = os.path.join(mother_path, "test_dfv2.csv")

loc_path = os.path.join(mother_path, "loc2popv2.pickle")



with open(train_path, 'rb') as f:

    train = pickle.load(f)



train = train[train.Location != 'Diamond Princess']



with open(test_path, 'rb') as f:

    test = pickle.load(f)    

    

with open(loc_path, 'rb') as f:

    loc2pop = pickle.load(f)

    



Locations = train.Location.unique()





'''

Design NN

'''



import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Concatenate

from tensorflow.keras.models import Model





tf.random.set_seed(0)

tf.keras.backend.set_floatx('float32')



      

class FC(Model):

    def __init__(self, input_dims):

        super(FC, self).__init__()

        self.input_dims = input_dims

        

        self.inputs = Input(shape=self.input_dims)

        self.dense1 = Dense(200, activation = 'softplus', name = 'dense1')

        self.subdense1 = Dense(50, activation = 'softplus', name = 'sub_dense1')

        self.subdense2 = Dense(1, activation = 'softplus', name = 'super_class')

        self.subdense3 = Dense(50, activation = 'softplus', name = 'sub_dense3')

        

        self.dense2 = Dense(50, activation = 'softplus', name = 'dense2')

       

        self.logit = Dense(1, activation ='softplus', name = 'logit')

        



    def build(self):

        x = self.inputs

        f = self.dense1(x)

        s  = self.subdense1(f)

        self.lat = self.subdense2(s)

        s = self.subdense3(self.lat)

        f = self.dense2(f)

        x = Concatenate()([f, s])

        logit = self.logit(x)

        return tf.keras.Model(inputs=self.inputs, outputs= [logit, self.lat])

        

def loss_fn(X, LOGITS):

    y, s = X

    y_logit, s_logit  = LOGITS

    y_loss = tf.keras.losses.MeanSquaredError()(y, y_logit)

    s_loss = tf.keras.losses.MeanSquaredError()(s, s_logit)

    return tf.reduce_mean(y_loss + s_loss)









period = 21



'''

make train dataset

'''

dic = {}

for loc in Locations:

    indexNames = train[train.Location == loc].index

    dic[loc] = np.array(train.loc[indexNames, "ConfirmedCases"])



trainset = []

testset = []



def allround(n):

    if n != 0:

        integer, decimal = int(np.log10(n)), np.log10(n) - int(np.log10(n))

        return int(10**(decimal)) * 10**integer

    else:

        return 0

    

weight_pop = 10000

def extract_feature(sig, pop):

    diff = np.diff(sig)

    diff = np.array([allround(n) for n in diff]).astype(np.float32)

    diff_ratio = weight_pop*diff/pop

    return diff_ratio

    

def target(sig, pop):

    diff = np.diff(sig)

    diff = np.array([allround(n) for n in diff]).astype(np.float32)

    diff_ratio = weight_pop*diff/pop

    return diff_ratio[0]

    

  

    

for i, loc in enumerate(Locations):

    if loc in ['Korea, South', 'China_Hubei']:

        loc_df =  dic[loc].copy()

        pop = loc2pop[loc]

        pop = allround(pop)

        try:

            k = np.where(loc_df != 0)[0][0]

        except:

            k = len(indexNames)-1

        try:

            critical_point = np.where(loc_df/pop>1e-07)[0][0]

        except:

            critical_point = len(indexNames)-1

        #Think here

        if k + period < len(indexNames)-1:

            # print(k)

            while k+period < len(indexNames)-1:

                tmp = loc_df[k:k+period+1]

                diff = extract_feature(tmp[:-1], pop)

                y_tmp = target(tmp[-2:], pop)

                days = 1/(np.maximum(1, k - critical_point-14))

                trainset.append([diff, days, y_tmp])

                k+=1

            tmp = loc_df[k:k+period+1]

            diff = extract_feature(tmp[:-1], pop)

            y_tmp = target(tmp[-2:], pop)

            days = 1/(np.maximum(1, k - critical_point-14))

            testset.append([diff, days, y_tmp])

            # print(k, feature4)

    

   





train_x = np.stack([trainset[i][0] for i in range(len(trainset))], axis = 0).astype(np.float32)

train_s = np.stack([trainset[i][1] for i in range(len(trainset))], axis = 0).astype(np.float32)

train_y = np.array([trainset[i][2] for i in range(len(trainset))]).astype(np.float32)

test_x = np.stack([testset[i][0] for i in range(len(testset))], axis = 0).astype(np.float32)

test_s = np.stack([testset[i][1] for i in range(len(testset))], axis = 0).astype(np.float32)

test_y = np.array([testset[i][2] for i in range(len(testset))]).astype(np.float32)



n_train_samples = len(train_x)

n_test_samples = len(test_x)



train_ds = tf.data.Dataset.from_tensor_slices(

    (train_x, train_y, train_s)).shuffle(n_train_samples).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_s)).batch(n_test_samples)



    

folder_name = os.path.join(mother_path, "mlp_p_{}".format(period))

try:

    os.mkdir(folder_name)

except OSError:

    pass  







lr = 0.01



model= FC(train_x.shape[1:]).build()    

            

compute_loss = loss_fn



optimizer = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')

test_loss = tf.keras.metrics.Mean(name='test_loss')





def train_step(inputs):

    X, Y, S = inputs

    with tf.GradientTape() as tape:

        logits = model(X)

        loss = compute_loss([Y, S], logits)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)  



def test_step(inputs):

    X, Y, S = inputs

    logits = model(X)

    loss = compute_loss([Y, S], logits)

    test_loss(loss)

    

    



def runs(log_freq = 1):

    for epoch in range(5000):

        for epoch_x, epoch_y, epoch_s in train_ds:

            inputs = [epoch_x, epoch_y, epoch_s]

            train_step(inputs)



        for epoch_x, epoch_y, epoch_s in test_ds:

            test_step([epoch_x, epoch_y, epoch_s])

        template = 'epoch: {}, train_loss: {}, test_loss: {}'

        print(template.format(epoch+1,

                                  train_loss.result(),

                                  test_loss.result()))    

        

# runs()





tf.keras.backend.clear_session()



'''

call confirmed forcast

'''

train_con_for_path = os.path.join(mother_path, "train_for_dfv2.csv")



with open(train_con_for_path, 'rb') as f:

    train_for = pickle.load(f)



dic_for = {}

for loc in Locations:

    indexNames = train_for[train_for.Location == loc].index

    dic_for[loc] = np.array(train_for.loc[indexNames, "c"])



dic_fat = {}

for loc in Locations:

    indexNames = train[train.Location == loc].index

    dic_fat[loc] = np.array(train.loc[indexNames, "Fatalities"])





trainset = []

testset = []

    

    



        

    

weight_con = 100

def extract_fat_feature(fat, con):

    diff = np.diff(fat)

    ratio = diff/con[:-1]

    ratio[np.where(con[:-1] == 0)[0]] = 0

    return weight_con*ratio

   

def target_fat(fat, con):

    diff = np.diff(fat)

    ratio = diff/con[:-1]

    ratio[np.where(con[:-1] == 0)[0]] = 0

    return weight_con*ratio[0]

    

  

    

for i, loc in enumerate(Locations):

    if loc in ['Korea, South', 'China_Hubei']:

        loc_df =  dic[loc].copy()

        loc_fat_df = dic_fat[loc].copy()

        pop = loc2pop[loc]

        pop = allround(pop)

        try:

            k = np.where(loc_df != 0)[0][0]

        except:

            k = len(indexNames)-1

        try:

            critical_point = np.where(loc_df/pop>1e-07)[0][0]

        except:

            critical_point = len(indexNames)-1

        #Think here

        if k + period < len(indexNames)-1:

            # print(k)

            while k+period < len(indexNames)-1:

                tmp_fat = loc_fat_df[k:k+period+1]

                tmp = loc_df[k:k+period+1]

                diff = extract_fat_feature(tmp_fat[:-1], tmp[:-1])

                y_tmp = target_fat(tmp_fat[-2:],tmp[-2:])

                days = 1/(np.maximum(1, k - critical_point-14))

                trainset.append([diff, days, y_tmp])

                k+=1

            tmp_fat = loc_fat_df[k:k+period+1]

            tmp = loc_df[k:k+period+1]

            diff = extract_fat_feature(tmp_fat[:-1], tmp[:-1])

            y_tmp = target_fat(tmp_fat[-2:],tmp[-2:])

            days = 1/(np.maximum(1, k - critical_point-14))

            testset.append([diff, days, y_tmp])

            # print(k, feature4)

    

   





train_x = np.stack([trainset[i][0] for i in range(len(trainset))], axis = 0).astype(np.float32)

train_s = np.stack([trainset[i][1] for i in range(len(trainset))], axis = 0).astype(np.float32)

train_y = np.array([trainset[i][2] for i in range(len(trainset))]).astype(np.float32)

test_x = np.stack([testset[i][0] for i in range(len(testset))], axis = 0).astype(np.float32)

test_s = np.stack([testset[i][1] for i in range(len(testset))], axis = 0).astype(np.float32)

test_y = np.array([testset[i][2] for i in range(len(testset))]).astype(np.float32)





n_train_samples = len(train_x)

n_test_samples = len(test_x)





train_ds = tf.data.Dataset.from_tensor_slices(

    (train_x, train_y, train_s)).shuffle(n_train_samples).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_s)).batch(n_test_samples)



    



folder_name = os.path.join(mother_path, "mlp_fat_p_{}".format(period))

try:

    os.mkdir(folder_name)

except OSError:

    pass  







lr = 0.005

model= FC(train_x.shape[1:]).build()    

            

compute_loss = loss_fn



optimizer = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

# optimizer = tf.keras.optimizers.Adam(lr=0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')

test_loss = tf.keras.metrics.Mean(name='test_loss')





def train_fat_step(inputs):

    X, Y, S = inputs

    with tf.GradientTape() as tape:

        logits = model(X)

        loss = compute_loss([Y, S], logits)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)  



def test_fat_step(inputs):

    X, Y, S = inputs

    logits = model(X)

    loss = compute_loss([Y, S], logits)

    test_loss(loss)

    

    





def runs_fat(log_freq = 1):

    for epoch in range(2):

        for epoch_x, epoch_y, epoch_s in train_ds:

            inputs = [epoch_x, epoch_y, epoch_s]

            train_fat_step(inputs)



        for epoch_x, epoch_y, epoch_s in test_ds:

            test_fat_step([epoch_x, epoch_y, epoch_s])

        template = 'epoch: {}, train_loss: {}, test_loss: {}'

        print(template.format(epoch+1,

                                  train_loss.result(),

                                  test_loss.result()))    

        



        

runs_fat()



mother_sub_path = "/kaggle/input/submission"

train_sub_path = os.path.join(mother_sub_path, "submission.csv")

submit = pd.read_csv(train_sub_path)





print(submit)

submit.to_csv('/kaggle/working/submission.csv',index=False)