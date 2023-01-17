# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/popularity-of-music-records/songs.csv',encoding='latin1')
data.head()
data.info()
data[data['year']==2010]
#How many songs does the dataset include for which the artist name is "Michael Jackson"?

data[data['artistname']== "Michael Jackson"]

#18 is the ans 
#Which of these songs by Michael Jackson made it to the Top 10? Select all that apply.

#Black or world

#rem the time

#in the closet

# you rock my world

#you are not alone
data.timesignature.unique()
data.timesignature.value_counts()
data.tempo.max()
data[data.tempo>=244]
songsTrain=data[data['year']<=2009]

songsTest=data[data['year']>=2010]
songsTrain.info()
X_train = songsTrain.drop(["Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)

Y_train = songsTrain.Top10



X_test = songsTest.drop(["Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)

Y_test = songsTest.Top10
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
model1.fit(X_train,Y_train)
yhat = model1.predict(X_train)

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

ll=log_loss(Y_train, yhat)
k=X_train.shape[1]
auc=2*k-2*ll
auc
ll
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

accuracy_score(Y_train, yhat)


data.corr()['Top10'][['timesignature_confidence', 'key_confidence', 'tempo_confidence']]
d = dict(zip(X_train, model1.coef_[0]))

[d[i] for i in ['timesignature_confidence', 'key_confidence', 'tempo_confidence']]
d
# 9 - Songs with heavier instrumentation tend to be louder (have higher values in the variable "loudness")

#     and more energetic (have higher values in the variable "energy").



#     By inspecting the coefficient of the variable "loudness", what does Model 1 suggest?





[d[i] for i in ['loudness', 'energy']]





#  Answer: positive coefficient for loudness -> so more chance to be in the top 10 (a hit)
X_train.corr()['loudness']['energy']
model2_train = songsTrain.drop(["loudness", "Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)

model2_test = songsTest.drop(["loudness", "Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)



model3_train = songsTrain.drop(["energy", "Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)

model3_test = songsTest.drop(["energy", "Top10", "year", "songtitle", "artistname", "songID", "artistID"], axis=1)
model1.fit(model2_train, Y_train)

dd = dict(zip(model2_train, model1.coef_[0]))

dd['energy']