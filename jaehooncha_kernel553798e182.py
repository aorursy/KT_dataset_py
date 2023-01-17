#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Apr 15 18:19:39 2020



@author: Jaehoon Cha



@email: chajaehoon79@gmail.com

"""

import pandas as pd

import numpy as np

import pickle

import os



'''

call data

'''

def make_data():   

    train_path = os.path.join("covid19-global-forecasting-week4", "train.csv")

    test_path = os.path.join("covid19-global-forecasting-week4", "test.csv")

    

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    

    

    train.columns.values[2] = 'Country'

    train.columns.values[1] = 'Subplace'

    train.fillna("", inplace = True)

    

    test.columns.values[2] = 'Country'

    test.columns.values[1] = 'Subplace'

    test.fillna("", inplace = True)

    

    with open('loc2popv4.pickle', 'rb') as f:

        loc2pop = pickle.load(f)





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

    for d in [6]:

        train.loc[(train.Location == 'Canada_Ontario') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 4540.0

    for d in [4]:

        train.loc[(train.Location == 'France_New Caledonia') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 18.0

    for d in [9]:

        train.loc[(train.Location == 'US_Arizona') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 3074.0

    for d in [22]:

        train.loc[(train.Location == 'US_District of Columbia') & (train.Date == '2020-03-{:02}'.format(d)), 'ConfirmedCases'] = 109.0

    for d in [2]:

        train.loc[(train.Location == 'US_New Hampshire') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 423.0

    for d in [9]:

        train.loc[(train.Location == 'US_New Mexico') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 991.0

    for d in [1]:

        train.loc[(train.Location == 'United Kingdom_Turks and Caicos Islands') & (train.Date == '2020-04-{:02}'.format(d)), 'ConfirmedCases'] = 5.0







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

    for d in [8]:

        train.loc[(train.Location == 'Canada_Prince Edward Island') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 0.0  

    for d in [5,6,7,8,9,10]:

        train.loc[(train.Location == 'Cyprus') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 11.0

    for d in [6]:

        train.loc[(train.Location == 'Finland') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 28.0

    for d in [4]:

        train.loc[(train.Location == 'Kazakhstan') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 6.0

    for d in [7]:

        train.loc[(train.Location == 'US_District of Columbia') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 24.0

    for d in [3]:

        train.loc[(train.Location == 'US_Montana') & (train.Date == '2020-04-{:02}'.format(d)), 'Fatalities'] = 6.0

       

       

    with open('train_dfv4.csv', 'wb') as f:

        pickle.dump(train, f)

    

    with open('test_dfv4.csv', 'wb') as f:

        pickle.dump(test, f)   



# make_data()



mother_path = "/kaggle/input/covid19week4"

train_path = os.path.join(mother_path, "train_dfv4.csv")

test_path = os.path.join(mother_path, "test_dfv4.csv")

loc_path = os.path.join(mother_path, "loc2popv4.pickle")

submit_path = os.path.join("/kaggle/input/covid19-global-forecasting-week-4", "submission.csv")



with open(train_path, 'rb') as f:

    train = pickle.load(f)



train = train[train.Location != 'Diamond Princess']

train = train[train.Location != 'MS Zaandam']





submit = pd.read_csv(submit_path)



with open(test_path, 'rb') as f:

    test = pickle.load(f)    

    

with open(loc_path, 'rb') as f:

    loc2pop = pickle.load(f)



Locations = train.Location.unique()





import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Concatenate

from tensorflow.keras.models import Model



tf.random.set_seed(0)

tf.keras.backend.set_floatx('float32')





batch_size = 32

total_epochs = 10000

epoch_log = 100

'''

make model

'''

      

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



def allround(n):

    if n > 0:

        integer, decimal = int(np.log10(n)), np.log10(n) - int(np.log10(n))

        return int(10**(decimal)) * 10**integer

    else:

        return 0







'''

confirmed case

'''



def run_confirmed():

    dic = {}

    for loc in Locations:

        indexNames = train[train.Location == loc].index

        dic[loc] = np.array(train.loc[indexNames, "ConfirmedCases"])

    



    #make trainset   

    trainset = []

    testset = []

    

    

    weight_pop = 100000

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

        

      

    period = 35

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

            if k + period < len(indexNames)-1:

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

        

       

    

    train_x = np.stack([trainset[i][0] for i in range(len(trainset))], axis = 0).astype(np.float32)

    train_s = np.stack([trainset[i][1] for i in range(len(trainset))], axis = 0).astype(np.float32)

    train_y = np.array([trainset[i][2] for i in range(len(trainset))]).astype(np.float32)

    test_x = np.stack([testset[i][0] for i in range(len(testset))], axis = 0).astype(np.float32)

    test_s = np.stack([testset[i][1] for i in range(len(testset))], axis = 0).astype(np.float32)

    test_y = np.array([testset[i][2] for i in range(len(testset))]).astype(np.float32)

    

    

    n_train_samples = len(train_x)

    n_test_samples = len(test_x)

    

        

    train_ds = tf.data.Dataset.from_tensor_slices(

        (train_x, train_y, train_s)).shuffle(n_train_samples).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_s)).batch(n_test_samples)







    model= FC(train_x.shape[1:]).build()    

                

    compute_loss = loss_fn

    

    optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    

    

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



    train_loss = tf.keras.metrics.Mean(name='train_loss')

    test_loss = tf.keras.metrics.Mean(name='test_loss')



    for epoch in range(total_epochs):

        for epoch_x, epoch_y, epoch_s in train_ds:

            inputs = [epoch_x, epoch_y, epoch_s]

            train_step(inputs)



        for epoch_x, epoch_y, epoch_s in test_ds:

            test_step([epoch_x, epoch_y, epoch_s])

        template = 'epoch: {}, train_loss: {}, test_loss: {}'

        if epoch % epoch_log == 0:

            print(template.format(epoch+1,

                                  train_loss.result(),

                                  test_loss.result()))    





    Result = []    

    for loc in Locations: 

        future = []

        loc_df =  dic[loc].copy()

        pop = loc2pop[loc]

        pop = allround(pop)

        try:

            critical_point = np.where(loc_df/pop>1e-07)[0][0]

        except:

            critical_point = len(indexNames)-1

        k = len(indexNames)-1-period

        #Think here

        tmp = loc_df[-period:]

        diff = extract_feature(tmp, pop)

        last = tmp[-1]

        for i in range(45):            

            days = 1/(np.maximum(1, k - critical_point-14))

            diff = np.expand_dims(diff, 0).astype(np.float32)

            y = model(diff)

            y_tmp = y[0].numpy()[0][0]

            last = int(pop*y_tmp/weight_pop) + last

            diff = np.append(diff[0][1:],[y_tmp])

            future.append(last)

        Result.append([loc,tmp,future])

        



    for i in range(len(Result)):

        tmp = np.append(Result[i][1][-13:], Result[i][2][:30])

        loc_idx = test[test.Location == Result[i][0]].index

        test.loc[loc_idx, 'ConfirmedCases'] = tmp     

         

    loc_idx = test[test.Location == 'Diamond Princess'].index

    test.loc[loc_idx, 'ConfirmedCases'] = 712



    loc_idx = test[test.Location == 'MS Zaandam'].index

    test.loc[loc_idx, 'ConfirmedCases'] = 9

    

                  

    return test   

                



    



test = run_confirmed()



tf.keras.backend.clear_session()



'''

fatalities

'''



def run_fatalities():

    dic_fat = {}

    for loc in Locations:

        indexNames = train[train.Location == loc].index

        dic_fat[loc] = np.array(train.loc[indexNames, "Fatalities"])



    dic = {}

    for loc in Locations:

        indexNames = train[train.Location == loc].index

        dic[loc] = np.array(train.loc[indexNames, "ConfirmedCases"])

    



    dic_for = {}

    for loc in Locations:

        indexNames = test[test.Location == loc].index

        dic_for[loc] = np.array(test.loc[indexNames, "ConfirmedCases"])

        

     

    trainset = []

    testset = []

    

    weight_con = 100

    period = 35



    def extract_feature(fat, con):

        diff = np.diff(fat)

        con_1 = np.array([allround(n) for n in con[:-1]]).astype(np.float32)

        ratio = diff/con_1

        ratio[np.where(con_1 == 0)[0]] = 0

        return weight_con*ratio

        

    def target(fat, con):

        diff = np.diff(fat)

        con_1 = np.array([allround(n) for n in con[:-1]]).astype(np.float32)

        ratio = diff/con_1

        ratio[np.where(con_1 == 0)[0]] = 0

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

                    diff = extract_feature(tmp_fat[:-1], tmp[:-1])

                    y_tmp = target(tmp_fat[-2:],tmp[-2:])

                    days = 1/(np.maximum(1, k - critical_point-14))

                    trainset.append([diff, days, y_tmp])

                    k+=1

                tmp_fat = loc_fat_df[k:k+period+1]

                tmp = loc_df[k:k+period+1]

                diff = extract_feature(tmp_fat[:-1], tmp[:-1])

                y_tmp = target(tmp_fat[-2:],tmp[-2:])

                days = 1/(np.maximum(1, k - critical_point-14))

                testset.append([diff, days, y_tmp])

    

    

    

    train_x = np.stack([trainset[i][0] for i in range(len(trainset))], axis = 0).astype(np.float32)

    train_s = np.stack([trainset[i][1] for i in range(len(trainset))], axis = 0).astype(np.float32)

    train_y = np.array([trainset[i][2] for i in range(len(trainset))]).astype(np.float32)

    test_x = np.stack([testset[i][0] for i in range(len(testset))], axis = 0).astype(np.float32)

    test_s = np.stack([testset[i][1] for i in range(len(testset))], axis = 0).astype(np.float32)

    test_y = np.array([testset[i][2] for i in range(len(testset))]).astype(np.float32)

    

    

    n_train_samples = len(train_x)

    n_test_samples = len(test_x)

    

    

    train_ds = tf.data.Dataset.from_tensor_slices(

        (train_x, train_y, train_s)).shuffle(n_train_samples).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_s)).batch(n_test_samples)

    

    model= FC(train_x.shape[1:]).build()    

                

    compute_loss = loss_fn

    

    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

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

        

        

    for epoch in range(total_epochs):

        for epoch_x, epoch_y, epoch_s in train_ds:

            inputs = [epoch_x, epoch_y, epoch_s]

            train_step(inputs)



        for epoch_x, epoch_y, epoch_s in test_ds:

            test_step([epoch_x, epoch_y, epoch_s])

        template = 'epoch: {}, train_loss: {}, test_loss: {}'

        if epoch % epoch_log == 0:

            print(template.format(epoch+1,

                                  train_loss.result(),

                                  test_loss.result()))    







    Result = []    

    for loc in Locations: 

        future = []

        loc_df =  dic[loc].copy()

        loc_fat_df = dic_fat[loc].copy()

        confromdf = dic_for[loc].copy()

        pop = loc2pop[loc]

        pop = allround(pop)

        try:

            critical_point = np.where(loc_df/pop>1e-07)[0][0]

        except:

            critical_point = len(indexNames)-1

        k = len(indexNames)-1-period

        tmp = loc_df[-period:]

        tmp_fat = loc_fat_df[-period:]

        diff = extract_feature(tmp_fat, tmp)

        last_fat = tmp_fat[-1]

        last = tmp[-1]

        con_idx = 13

        for i in range(30):            

            days = 1/(np.maximum(1, k - critical_point-14))

            diff = np.expand_dims(diff, 0).astype(np.float32)

            y = model(diff)

            y_tmp = y[0].numpy()[0][0]

            last_fat = int(last*y_tmp/weight_con) + last_fat

            diff = np.append(diff[0][1:],[y_tmp])

            future.append(last_fat)

            last = confromdf[con_idx]

            con_idx += 1

        Result.append([loc,tmp_fat,future])



    for i in range(len(Result)):

        tmp = np.append(Result[i][1][-13:], Result[i][2][:30])

        loc_idx = test[test.Location == Result[i][0]].index

        test.loc[loc_idx, 'Fatalities'] = tmp



    loc_idx = test[test.Location == 'Diamond Princess'].index

    test.loc[loc_idx, 'Fatalities'] = 11



    loc_idx = test[test.Location == 'MS Zaandam'].index

    test.loc[loc_idx, 'Fatalities'] = 2



    return test   

    

                  

test = run_fatalities()

tf.keras.backend.clear_session()



submit.ConfirmedCases = test.ConfirmedCases

submit.Fatalities = test.Fatalities



submit.to_csv('/kaggle/working/submission.csv',index=False)



print('complete')










