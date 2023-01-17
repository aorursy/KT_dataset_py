# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder

seed = 7

np.random.seed(seed)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

features = df.columns

data = df.drop(["id","type"], axis=1)

labels = df["type"]
model = Sequential()

model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))

model.add(Dense(5, init='uniform', activation='relu'))

model.add(Dense(3, init='uniform', activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#temp = data['color']

#encoder = LabelEncoder()

#encoder.fit(temp)

#l = encoder.transform(temp)

#data = data.drop(['color'], axis=1)

#data = np.hstack((np.array(data),l))



print(np.concatenate((data,l), axis=0))

model.fit(data, labels, nb_epoch=150, batch_size=10)