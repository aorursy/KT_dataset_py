#This code comes with ABSOLUTELY NO WARRANTY

class GaussianObservation:

        def __init__(self, filename):

                import pandas as pd

                with pd.HDFStore(filename, "r") as train:

                    self.trainh5= train.get("train")

                

        def reset(self):

                self.train = self.trainh5[self.trainh5["timestamp"]<=905]

                valid = self.trainh5[self.trainh5["timestamp"]>905]

                yval = valid['y']

                yval.reset_index(drop=True,inplace=True)

                yval = ((yval - yval.mean())**2).sum()

                self.ssq=yval

                self.validgroupby = valid.groupby("timestamp")

                self.features = self.validgroupby.get_group(906)

                self.features = self.features.iloc[:,0:109]

                self.target = self.validgroupby.get_group(906)[['id','y']]

                self.target['y'] = 0.0;

                self.features.reset_index(drop=True,inplace=True)

                self.target.reset_index(drop=True,inplace=True)

                self.currenttime=906;

                self.sse = 0

                self.done = False

                self.info = {}

                return self



        def step(self,target):

                import math

                if self.done:

                       return self, math.nan, True, self.info

                pred = target['y']

                yval = self.validgroupby.get_group(self.currenttime)['y']

                yval.reset_index(drop=True,inplace=True)

                dnom = ((yval - yval.mean())**2).sum()

                num = ((yval - pred)**2).sum()

                self.sse=self.sse+num

                r2 = 1-num/dnom

                reward = math.sqrt(math.fabs(r2))

                if r2<0:

                        reward = -reward

                self.train=None

                if self.currenttime<1812:

                        self.features = self.validgroupby.get_group(self.currenttime+1)

                        self.features = self.features.iloc[:,0:109]

                        self.target = self.validgroupby.get_group(self.currenttime+1)[['id','y']]

                        self.target['y'] = 0.0;

                        self.features.reset_index(drop=True,inplace=True)

                        self.target.reset_index(drop=True,inplace=True)

                else:

                        self.done = True

                self.currenttime=self.currenttime+1

                self.info = {}

                if self.done:

                        r2 = 1-self.sse/self.ssq

                        score = math.sqrt(math.fabs(r2))

                        if r2<0:

                                score = -score

                        self.info['public_score']=score

                return self, reward, self.done, self.info
#env = kagglegym.make()

env = GaussianObservation("../input/train.h5")

observation = env.reset()

while True:

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    observation, reward, done, info = env.step(target)

    if timestamp % 100 == 0:

        print("Timestamp {0} reward {1}".format(timestamp,reward))

    if done:

        print("Public score: {}".format(info["public_score"]))

        break



#kagglegym numbers

print("===================================")

import kagglegym

import numpy as np

import pandas as pd

env = kagglegym.make()

observation = env.reset()

while True:

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    observation, reward, done, info = env.step(target)

    if timestamp % 100 == 0:

        print("Timestamp {0} reward {1}".format(timestamp,reward))

    if done:

        print("Public score: {}".format(info["public_score"]))

        break
