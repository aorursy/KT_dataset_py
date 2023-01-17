# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")

test = pd.read_csv("../input/test.csv")

import matplotlib.pyplot as plt

import seaborn as sns

corrmat = train.corr()

cols = corrmat.nlargest(12, 'target')['target'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
target = train ['target']

features = train.drop(['target','ID_code'], axis =1)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features,target,test_size = 0.1,random_state =0)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(100, input_dim=200, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='sigmoid'))



model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train, epochs=10, batch_size=30)



scores = model.evaluate(x_test, y_test)
scores
t = test.drop(['ID_code'], axis=1)

prediction = model.predict(t)

pdf=[]

for i in range (len(prediction)):

    if (prediction[i][0]>0.5):

        pdf.append(1)

    else :

        pdf.append(0)



output = pd.DataFrame({'ID_code': test['ID_code'],'target':pdf})

output.to_csv('submission.csv', index=False)