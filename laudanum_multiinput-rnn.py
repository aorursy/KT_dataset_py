import os

import numpy as np

import pandas as pd

import sqlite3

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop

from IPython.display import display

from sklearn import preprocessing
connection = sqlite3.connect("../input/database.sqlite")

cursor = connection.cursor()

bundesliga = pd.read_sql_query("SELECT * FROM FlatView_Advanced", connection)

cursor.close()

connection.close()
#adapt data types and get rid of 2017 matches as matches still lie in the future (as of mid 2017)

bundesliga.Season = bundesliga.Season.astype(np.int64)

bundesliga.KaderHome = bundesliga.KaderHome.astype(np.float64)

bundesliga.AvgAgeHome = bundesliga.AvgAgeHome.astype(np.float64)

bundesliga.ForeignPlayersHome = bundesliga.ForeignPlayersHome.astype(np.float64)

bundesliga.OverallMarketValueHome = bundesliga.OverallMarketValueHome.astype(np.float64)

bundesliga.AvgMarketValueHome = bundesliga.AvgMarketValueHome.astype(np.float64)

bundesliga.StadiumCapacity = bundesliga.StadiumCapacity.astype(np.float64)

bundesliga.FTHG = bundesliga.FTHG.astype(np.float64)

bundesliga.FTAG = bundesliga.FTAG.astype(np.float64)





bundesliga = bundesliga[bundesliga.Season<2017]
bundesliga.loc[bundesliga.HomeTeam == bundesliga.Unique_Team,"UniqueHome"] = "1"

bundesliga.loc[bundesliga.HomeTeam != bundesliga.Unique_Team,"UniqueHome"] = "0"



pd.options.mode.chained_assignment = None  # default='warn'

bundesliga.loc[(bundesliga.HomeTeam==bundesliga.Unique_Team) & (bundesliga.FTR=="H") ,("WDL")] = "2"

bundesliga.loc[bundesliga.FTR=="D" ,"WDL"] = "1"

bundesliga.loc[(bundesliga.HomeTeam==bundesliga.Unique_Team) & (bundesliga.FTR=="A") ,"WDL"] = "0"

bundesliga.loc[(bundesliga.AwayTeam==bundesliga.Unique_Team) & (bundesliga.FTR=="H") ,("WDL")] = "0"

bundesliga.loc[(bundesliga.AwayTeam==bundesliga.Unique_Team) & (bundesliga.FTR=="D") ,"WDL"] = "1"

bundesliga.loc[(bundesliga.AwayTeam==bundesliga.Unique_Team) & (bundesliga.FTR=="A") ,"WDL"] = "2"
le = preprocessing.LabelEncoder()

le.fit(bundesliga.FTR.values)

bundesliga.FTR = le.transform(bundesliga.FTR)



FTR_dummies = pd.get_dummies(bundesliga.FTR)

WDL_dummies = pd.get_dummies(bundesliga.WDL)



FTR_dummies.columns = ['A','D1','H']

WDL_dummies.columns = ['W','D2','L']



bundesliga = pd.concat([bundesliga,FTR_dummies],axis=1)

bundesliga = pd.concat([bundesliga,WDL_dummies],axis=1)



bundesliga = bundesliga.drop('WDL', 1)

bundesliga = bundesliga.drop('FTR', 1)

train_data = bundesliga[bundesliga.Season < 2015]

val_data = bundesliga[bundesliga.Season == 2015]

test_data = bundesliga[bundesliga.Season == 2016] 



#reset indexes to allow easy iteration over data frames

train_data = train_data.reset_index()

val_data = val_data.reset_index()

test_data = test_data.reset_index()
#Get mean of training set and use for validation set in order not to introduce future knowledge into training set

mean = train_data.select_dtypes(include=[np.float]).mean(axis=0) 



train_data.loc[:,train_data.select_dtypes(include=[np.float]).columns.values] -= mean

val_data.loc[:,val_data.select_dtypes(include=[np.float]).columns.values] -= mean

test_data.loc[:,test_data.select_dtypes(include=[np.float]).columns.values] -= mean



std = train_data.select_dtypes(include=[np.float]).std(axis=0)

train_data.loc[:,train_data.select_dtypes(include=[np.float]).columns.values] /= std

val_data.loc[:,val_data.select_dtypes(include=[np.float]).columns.values] /= std

test_data.loc[:,test_data.select_dtypes(include=[np.float]).columns.values] /= std
print("Train Win: "+str(sum(train_data.loc[:,'H'])/len(train_data[:])))

print("Train Draw: "+str(sum(train_data.loc[:,'D1'])/len(train_data[:])))

print("Train Lose: "+str(sum(train_data.loc[:,'A'])/len(train_data[:])))



print("Val Win: "+str(sum(val_data.loc[:,'H'])/len(val_data[:])))

print("Val Draw: "+str(sum(val_data.loc[:,'D1'])/len(val_data[:])))

print("Val Lose: "+str(sum(val_data.loc[:,'A'])/len(val_data[:])))



print("Test Win: "+str(sum(test_data.loc[:,'H'])/len(test_data[:])))

print("Test Draw: "+str(sum(test_data.loc[:,'D1'])/len(test_data[:])))

print("Test Lose: "+str(sum(test_data.loc[:,'A'])/len(test_data[:])))



print("Overall Win: "+str(sum(bundesliga.H)/len(bundesliga[:])))

print("Overall Draw: "+str(sum(bundesliga.D1)/len(bundesliga[:])))

print("Overall Lose: "+str(sum(bundesliga.A)/len(bundesliga[:])))







def generator(data, lookback, delay,batch_size=10, step=1):

    

    i = lookback

    #matches = data.iloc[lookback::step]

   

    while 1:

        

        target_1_np = []

        target_2_np = []

        samples_1_np = []

        samples_2_np = []

        samples_meta_np = []

        

        #for loop in order to fill batch of len(batch_size) with examples

        for batch_no in range(batch_size):

            #Make sure index does not exceed length of array

            if i+batch_size > len(data)-1:

                i = lookback



            #Make sure team history does not include matches of other teams

            while data.loc[i].Unique_Team_ID != data.loc[i-lookback].Unique_Team_ID:

                i+=1 

            

            #Retrieve match to be predicted

            target_game = data.loc[i]

            target_1 = data.loc[i][["A","D1","H"]]

            start_idx = i-lookback

            end_idx = i

            #Retrieve history for current "unique_team"

            samples_1 = data.iloc[start_idx:end_idx][['L','D1','W','FTHG','FTAG']]

            match_id = target_game['Match_ID']

            #Retrieve additional meta data for "unique team's" matches and match to be predicted

            samples_meta_1 = data.iloc[start_idx+1:end_idx+1][['KaderHome','AvgAgeHome','ForeignPlayersHome','OverallMarketValueHome','AvgMarketValueHome','StadiumCapacity','UniqueHome']]

            #Retrieve opposing team

            other_team = target_game['AwayTeam'] if target_game['Unique_Team'] == target_game['HomeTeam'] else target_game['HomeTeam']

            #Retrieve row where match_ID equals target_1.Match_ID but within opponent teams schedule

            target_2 = data[(data['Match_ID'] == target_game['Match_ID']) & (data['Unique_Team'] == other_team)]

            target_2 = target_2[["L","D1","W"]]

            match_idx = np.where((data['Match_ID'] == target_game['Match_ID']) & (data['Unique_Team'] == other_team))[0][0]

            #Retrieve history for opposing team

            samples_2 = data.iloc[match_idx-lookback:match_idx,][['L','D1','W','FTHG','FTAG']]

            #Retrieve meta_information to be supplied for opposing team -> should be fixed as currently always holds data only for teams taking part in match to be predicted (not for history)!!! 

            samples_meta_2 = data.iloc[match_idx-lookback+1:match_idx+1][['KaderHome','AvgAgeHome','ForeignPlayersHome','OverallMarketValueHome','AvgMarketValueHome','StadiumCapacity']]

            samples_meta_2.columns = ['KaderTeam2','AvgAgeTeam2','ForeignPlayerTeam2','OverallMarketValueTeam2','AvgMarketValueTeam2','StadiumCapacityTeam2']

            

            #add targets and samples to batch

            target_1_np.append(target_1.values)

            samples_1_np.append(samples_1.values)

            target_2_np.append(target_2.values)

            samples_2_np.append(samples_2.values)

            

            samples_meta_np.append(pd.concat([samples_meta_1.reset_index(drop=True), samples_meta_2.reset_index(drop=True)], axis=1).values)

            i += 1

            

            #turn pandas data frames into numpy arrays

            target_1_ar = np.array(target_1_np)

            samples_1_ar = np.array(samples_1_np)

            samples_2_ar = np.array(samples_2_np)

            samples_meta_ar = np.array(samples_meta_np)

           

    

        yield [samples_1_ar,samples_2_ar,samples_meta_ar],target_1_ar 



#How many matches in the past shall be considered?

lookback = 10

#Consider every step's match for prediction

step = 1

#Predict the delay's next match (0 = next match)

delay = 0

#how many samples shall be presented to the NN per step?

batch_size = 10



#create generators for training, validation and test data

train_gen = generator(train_data, lookback=lookback, delay=delay, step=step,

batch_size=batch_size)

val_gen = generator(val_data, lookback=lookback, delay=delay,step=step, batch_size=batch_size)

test_gen = generator(test_data, lookback=lookback, delay=delay, step=step, batch_size=batch_size)



# This is how many steps to draw from `val_gen` in order to see the whole validation set:

val_steps = (len(val_data)) - (lookback) // batch_size



# This is how many steps to draw from `test_gen` in order to see the whole test set:

test_steps = (len(test_data)) - (lookback) // batch_size
from keras.models import Model

from keras import layers

from keras import Input

from keras.layers import Flatten



#RNN1: Input is match history of team 1

RNN1_input = Input(shape=(None, 5), name='RNN1')

RNN1_GRU = layers.GRU(128,activation="relu")(RNN1_input)



#RNN2: Input is match history of team 2

RNN2_input = Input(shape=(None, 5), name='RNN2')

RNN2_GRU = layers.GRU(128,activation="relu")(RNN2_input)



#Dense: Input are metadata for team 1 and 2 (Currently not including history -> Which is most likely decreasing accuracy significantly)

Meta_Input = Input(shape=(batch_size, 13), name='meta')

Meta_out = layers.Dense(128,activation="relu")(Meta_Input)

Meta_out = Flatten()(Meta_out)



#Concatenate 3 Input layers

concatenated = layers.concatenate([RNN1_GRU,RNN2_GRU,Meta_out])



#Add Dense layers on top for deep feature recognition

concat_learn = layers.Dense(64,activation='relu')(concatenated)

concat_learn2 = layers.Dense(32,activation='relu')(concat_learn)



#Add a softmax classifier on top

answer = layers.Dense(3, activation='softmax')(concat_learn2)



# At model instantiation, specify the two inputs and the output:

model = Model([RNN1_input, RNN2_input, Meta_Input], answer)

model.summary()



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['acc'])



#change epochs and steps per epoch as you see fit

history = model.fit_generator(train_gen,

                              steps_per_epoch=500,

                              epochs=5,

                              validation_data=val_gen,

                              validation_steps=val_steps)
import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['acc']

val_acc = history.history['val_acc']



epochs = range(len(acc))



plt.figure()



plt.plot(epochs, acc, 'b+')

plt.plot(epochs, val_acc, 'bo')

plt.title('Training and validation accuracy')
import matplotlib.pyplot as plt

%matplotlib inline



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.figure()



plt.plot(epochs, loss, 'r+')

plt.plot(epochs, val_loss, 'ro')

plt.title('Training and validation loss')