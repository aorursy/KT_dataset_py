import math

import pandas as pd

import numpy as np        

        

class FakeGym():

    def __init__(self, df):

        self._df=df

        self._current_ts=math.floor(df["timestamp"].max()/2)+1

        self._max_ts=df["timestamp"].max()

        self._R=-1

        self._predicted=[]

        self._actual=[]

        

        self._done=False

        self._info={}

        

        

        self.train=df[df["timestamp"]<self._current_ts]

        self.target=df[df["timestamp"]==self._current_ts][["id","y"]]

        self.target.loc[:,"y"]=0

        self.target=self.target.reset_index()[["id","y"]]

        

        col_list=df.columns.tolist()

        col_list.pop()

        self._col_list=col_list

        self.features=self._df[self._df["timestamp"]==self._current_ts][self._col_list]

        self.features=self.features.reset_index()[self._col_list]

        

    def step(self,target):

        self.train=None

        actual_target=self._df.loc[self._df["timestamp"]==self._current_ts,["id","y"]]

        self._predicted += target["y"].tolist()

        self._actual += actual_target["y"].tolist()

        #R calculation

        if observation._current_ts % 100 == 0 or self._current_ts == self._max_ts:

            s_actual=pd.Series(self._actual)

            s_predicted=pd.Series(self._predicted)

            ybar=s_actual.mean()

            self._R = 1-np.sum((s_predicted-s_actual)**2)/np.sum((ybar-s_actual)**2)

            self._R = np.sign(self._R)*(abs(self._R)**0.5)



        self._current_ts += 1

        if self._current_ts > self._max_ts:

            self._done=True

            self._info["public_score"]=self._R

        else:

            self.target=self._df.loc[df["timestamp"]==self._current_ts,["id","y"]]

            self.target.loc[:,"y"]=0

            self.target=self.target.reset_index()[["id","y"]]

            self.features=self._df.loc[self._df["timestamp"]==self._current_ts,self._col_list]

            self.features=self.features.reset_index()[self._col_list]

        return self, self._R, self._done, self._info

            

            

            
with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")

    

observation = FakeGym(df) #different



# Note that the first observation we get has a "train" dataframe

print("Train has {} rows".format(len(observation.train)))



# The "target" dataframe is a template for what we need to predict:

print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))



while True:

    timestamp = observation.features["timestamp"][0]

    target = observation.target

    # We perform a "step" by making our prediction and getting back an updated "observation":

    observation, reward, done, info = observation.step(target) #different

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        print("Reward {}".format(reward))

    if done:

        print("Public score: {}".format(info["public_score"]))

        break

        

#Manual check all predicted=0

actual=df["y"][(df["timestamp"] >= 907)]

actual=actual.reset_index()["y"]

num = np.sum(actual**2)

ybar=actual.mean()

den = np.sum((actual-ybar)**2)

R2=1-num/den

R=np.sign(R2)*(abs(R2)**0.5)

print("Manually checked public score: {}".format(R))
import kagglegym

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# Note that the first observation we get has a "train" dataframe

print("Train has {} rows".format(len(observation.train)))



# The "target" dataframe is a template for what we need to predict:

print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))



while True:

    timestamp = observation.features["timestamp"][0]

    target = observation.target

    # We perform a "step" by making our prediction and getting back an updated "observation":

    observation, reward, done, info = env.step(target)

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

        print("Reward {}".format(reward))



    

    if done:

        print("Public score: {}".format(info["public_score"]))

        break