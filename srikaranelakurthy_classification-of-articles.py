# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

missing_values = ["n/a", "na", "--","-"]

newsdat=pd.read_csv("/kaggle/input/online-news-popularity/OnlineNewsPopularity.csv",na_values = missing_values)
newsdat.shape
newsdat.head(5)
lifestyledf=newsdat[newsdat[' data_channel_is_lifestyle']==1]
lifestyledf['dat_ch']='lifestyle'
lifestyledf.shape
entertainmentdf=newsdat[newsdat[' data_channel_is_entertainment']==1]
entertainmentdf['dat_ch']='entertainment'
entertainmentdf.shape
busdf=newsdat[newsdat[' data_channel_is_bus']==1]
busdf['dat_ch']='bus'
busdf.shape
socmeddf=newsdat[newsdat[' data_channel_is_socmed']==1]
socmeddf['dat_ch']='socmed'
socmeddf.shape
techdf=newsdat[newsdat[' data_channel_is_tech']==1]
techdf['dat_ch']='tech'
techdf.shape
worlddf=newsdat[newsdat[' data_channel_is_world']==1]
worlddf['dat_ch']='world'
worlddf.shape
nonclasdf=newsdat[(newsdat[' data_channel_is_world']==0) & (newsdat[' data_channel_is_tech']==0) & (newsdat[' data_channel_is_socmed']==0) & (newsdat[' data_channel_is_bus']==0) & (newsdat[' data_channel_is_lifestyle']==0) & (newsdat[' data_channel_is_entertainment']==0)]
nonclasdf['dat_ch']='other'
nonclasdf.shape
frames=[nonclasdf,worlddf,techdf,socmeddf,busdf,lifestyledf,entertainmentdf]

newsdatcr=pd.concat(frames,ignore_index=True)

newsdatcr=newsdatcr.drop(['url',' data_channel_is_lifestyle',' data_channel_is_entertainment',' data_channel_is_bus',' data_channel_is_socmed',' data_channel_is_tech',' data_channel_is_world'],axis=1)
plt.figure(figsize=(40,30))

cor = newsdatcr.corr(method ='pearson')

sns.heatmap(cor, cmap="RdYlGn")

plt.show()
newsdatcr=newsdatcr.drop([' n_non_stop_words',' n_unique_tokens',' kw_avg_min',' kw_avg_avg',' self_reference_avg_sharess'],axis=1)
newsdatcr.to_csv(r'Updatednewspopularity.csv')

X1= newsdatcr.iloc[:,0:49]

y1= newsdatcr.iloc[:,49]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X1 = sc.fit_transform(X1)

Y1=y1.values

encoder = LabelEncoder()
newsdatcr.isnull().any()
encoder.fit(Y1)
encoded_Y1 = encoder.transform(Y1)

transf_y1 = np_utils.to_categorical(encoded_Y1)

transf_y1
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, transf_y1, test_size=0.3)
model = Sequential()

model.add(Dense(800, input_dim=49, activation='relu'))

model.add(Dense(600, activation='relu'))

model.add(Dense(400, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(200, activation='relu'))

model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=80, batch_size=32, verbose=1)
eval_model=model.evaluate(X_train, y_train)

eval_model
y_pred=model.predict(X_test)

#Converting predictions to label

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

#Converting one hot encoded test label to label

test = list()

for i in range(len(y_test)):

    test.append(np.argmax(y_test[i]))
from sklearn.metrics import accuracy_score

a = accuracy_score(pred,test)

print('Accuracy is:', a*100)
from keras.utils import plot_model

plot_model(model, to_file='model.png')