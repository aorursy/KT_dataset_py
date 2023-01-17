import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from IPython.display import display, Image

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline
voice_data = pd.read_csv('../input/voice.csv')
#Show the features

voice_data.columns
voice_data.loc[voice_data['label'] == 'male', 'label'] = 0

voice_data.loc[voice_data['label'] == 'female', 'label'] = 1

# Commented code shows

# Male Data IndexRange from 0 - 1583

# Female Data IndexRange from 1584 - 3167
# Grab and Seperate Male and female observations

X_male = voice_data.loc[voice_data['label'] == 0, 'label'].index.tolist()

X_female = voice_data.loc[voice_data['label'] == 1, 'label'].index.tolist()
# Visualize some of the features which you expect to be most distinguishing



sd = voice_data['sd']

male_voicedata = voice_data['meanfreq'][X_male]

female_voicedata = voice_data['meanfreq'][X_female]



# Mean Frequency Plots

feature = 'meanfreq'

fig_meanfreq = plt.figure(figsize=(7,5))

ax = fig_meanfreq.add_subplot(111)

bp = ax.boxplot([voice_data[feature]*1000, voice_data[feature][X_male]*1000])

ax.set_xticklabels(['Female', 'Male'])

plt.title(feature)

plt.ylabel('Frequency [Hz]')



# Mean Fundamental Frequency Plots

feature = 'meanfun'

fig_meanfreq = plt.figure(figsize=(7,5))

ax = fig_meanfreq.add_subplot(111)

bp = ax.boxplot([voice_data[feature]*1000, voice_data[feature][X_male]*1000])

ax.set_xticklabels(['Female', 'Male'])

plt.title(feature)

plt.ylabel('Frequency [Hz]')





# Mean Dominant Frequency Plots

feature = 'meandom'

fig_meanfreq = plt.figure(figsize=(7,5))

ax = fig_meanfreq.add_subplot(111)

bp = ax.boxplot([voice_data[feature]*1000, voice_data[feature][X_male]*1000])

ax.set_xticklabels(['Female', 'Male'])

plt.title(feature)

plt.ylabel('Frequency [Hz]')





# Mean Dominant Frequency Plots

feature = 'dfrange'

fig_meanfreq = plt.figure(figsize=(7,5))

ax = fig_meanfreq.add_subplot(111)

bp = ax.boxplot([voice_data[feature]*1000, voice_data[feature][X_male]*1000])

ax.set_xticklabels(['Female', 'Male'])

plt.title(feature)

plt.ylabel('Frequency [Hz]')





print('male median frequency', '\t\tfemale median frequency')

print(male_voicedata.median()*1000,'\t\t', female_voicedata.median()*1000)
# We see visually that the mean fundamental frequency provides the clearest signal between male and female

# for the features examined
# Train a random forrest classifier using cross validation

# Use all predictors



# If we predict on fudamental frequency alone we can achieve ~ 95% accuaracy

#predictors = ['meanfun']



# Using all features we can achieve ~ 98% accuaracy

predictors = [

              'meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',

              'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',

              'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'

             ]





alg = RandomForestClassifier( n_estimators=35, min_samples_split=3, min_samples_leaf=5)

kf = cross_validation.KFold(voice_data.shape[0],n_folds=10, shuffle=True)

scores = cross_validation.cross_val_score(alg, voice_data[predictors], list(voice_data['label']), cv = kf)

print(scores.mean())