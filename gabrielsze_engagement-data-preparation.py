import pandas as pd

import random

import numpy as np
random.seed(1)



def _fillHighValues(x):

    return random.randint(8,10)



def _fillMidValues(x):

    return random.randint(4,7)



def _fillLowValues(x):

    return random.randint(1,3)



def _fillNoise(x):

    return random.randint(1,10)



def _engagedHigh(x):

    return random.randint(4,5)



def _engagedMid(x):

    return random.randint(2,3)



def _engagedLow(x):

    return random.randint(1,1)



def _engagedNoise(x):

    return random.randint(1,5)
df = pd.DataFrame(index = range(0,3000), columns = ['Emotion', 'HeadGaze', 'Motion', 'handPose', 'sleepPose', 'Engagement'])
#First 80 rows are synthetically created, last 20 rows will be noise



df['Emotion'][:900] = df['Emotion'][:900].apply(lambda x : _fillHighValues(x))

df['Emotion'][900:1800] = df['Emotion'][900:1800].apply(lambda x : _fillMidValues(x))

df['Emotion'][1800:2700] = df['Emotion'][1800:2700].apply(lambda x : _fillLowValues(x))

df['Emotion'][2700:] = df['Emotion'][2700:].apply(lambda x : _fillNoise(x))



df['HeadGaze'][:900] = df['HeadGaze'][:900].apply(lambda x : _fillHighValues(x))

df['HeadGaze'][900:1800] = df['HeadGaze'][900:1800].apply(lambda x : _fillMidValues(x))

df['HeadGaze'][1800:2700] = df['HeadGaze'][1800:2700].apply(lambda x : _fillLowValues(x))

df['HeadGaze'][2700:] = df['HeadGaze'][2700:].apply(lambda x : _fillNoise(x))



df['Motion'][:900] = df['Motion'][:900].apply(lambda x: _fillHighValues(x))

df['Motion'][900:1800] = df['Motion'][900:1800].apply(lambda x: _fillMidValues(x))

df['Motion'][1800:2700] = df['Motion'][1800:2700].apply(lambda x: _fillLowValues(x))

df['Motion'][2700:] = df['Motion'][2700:].apply(lambda x: _fillNoise(x))



df['handPose'][:900] = df['handPose'][:900].apply(lambda x: _fillHighValues(x))

df['handPose'][900:1800] = df['handPose'][900:1800].apply(lambda x: _fillMidValues(x))

df['handPose'][1800:2700] = df['handPose'][1800:2700].apply(lambda x: _fillLowValues(x))

df['handPose'][2700:] = df['handPose'][2700:].apply(lambda x: _fillNoise(x))



df['sleepPose'][:900] = df['sleepPose'][:900].apply(lambda x: _fillLowValues(x))

df['sleepPose'][900:1800] = df['sleepPose'][900:1800].apply(lambda x: _fillMidValues(x))

df['sleepPose'][1800:2700] = df['sleepPose'][1800:2700].apply(lambda x: _fillHighValues(x))

df['sleepPose'][2700:] = df['sleepPose'][2700:].apply(lambda x: _fillNoise(x))



df['Engagement'][:900] = df['Engagement'][:900].apply(lambda x : _engagedHigh(x))

df['Engagement'][900:1800] = df['Engagement'][900:1800].apply(lambda x : _engagedMid(x))

df['Engagement'][1800:2700] = df['Engagement'][1800:2700].apply(lambda x : _engagedLow(x))

df['Engagement'][2700:] = df['Engagement'][2700:].apply(lambda x : _engagedNoise(x))
df.describe()
df.head()
df.to_excel('EngagementTest3.xlsx')