# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegressionCV





df = pd.read_csv("../input/user1.features_labels.csv")

dropitall = df[['audio_naive:mfcc0:mean','audio_naive:mfcc1:mean','audio_naive:mfcc2:mean','audio_naive:mfcc3:mean','audio_naive:mfcc4:mean','audio_naive:mfcc5:mean','audio_naive:mfcc6:mean','audio_naive:mfcc7:mean','audio_naive:mfcc8:mean','audio_naive:mfcc9:mean','audio_naive:mfcc10:mean','audio_naive:mfcc11:mean','audio_naive:mfcc12:mean','label:STROLLING','label:DRINKING__ALCOHOL_','label:TOILET','label:TALKING','label:SURFING_THE_INTERNET','label:WITH_CO-WORKERS','label:WITH_FRIENDS']].dropna(axis=0, how='any')

#selfreportedonly = df[['label:STROLLING','label:DRINKING__ALCOHOL_','label:TOILET','label:TALKING','label:SURFING_THE_INTERNET','label:WITH_CO-WORKERS','label:WITH_FRIENDS']]

#X_audiomeans = df[['audio_naive:mfcc0:mean','audio_naive:mfcc1:mean','audio_naive:mfcc2:mean','audio_naive:mfcc3:mean','audio_naive:mfcc4:mean','audio_naive:mfcc5:mean','audio_naive:mfcc6:mean','audio_naive:mfcc7:mean','audio_naive:mfcc8:mean','audio_naive:mfcc9:mean','audio_naive:mfcc10:mean','audio_naive:mfcc11:mean','audio_naive:mfcc12:mean']]



#print(dropitall)

X = dropitall[['audio_naive:mfcc0:mean','audio_naive:mfcc1:mean','audio_naive:mfcc2:mean','audio_naive:mfcc3:mean','audio_naive:mfcc4:mean','audio_naive:mfcc5:mean','audio_naive:mfcc6:mean','audio_naive:mfcc7:mean','audio_naive:mfcc8:mean','audio_naive:mfcc9:mean','audio_naive:mfcc10:mean','audio_naive:mfcc11:mean','audio_naive:mfcc12:mean']]

y = dropitall[['label:DRINKING__ALCOHOL_']]



searchCV = LogisticRegressionCV(

    Cs=list(np.power(10.0, np.arange(-10, 10)))

    ,penalty='l2'

    ,scoring='roc_auc'

    ,cv=5

    ,random_state=777

    ,max_iter=10000

    ,fit_intercept=True

    ,solver='newton-cg'

    ,tol=10

)

searchCV.fit(X, y)

print ('Max auc_roc:', searchCV.scores_[1].max())



searchCV.score(X, dropitall[['label:STROLLING']])

searchCV.get_params()
