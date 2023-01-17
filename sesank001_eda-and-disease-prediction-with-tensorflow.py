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
df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv')
df.head()
df.columns.values
columns = (df.columns.values[0]).split(';')
new_df_dict = {}



for index, row in df.iterrows():

    lst = str(row.values[0]).split(';')

    new_df_dict[index]=lst[1:]
data = pd.DataFrame.from_dict(new_df_dict, orient='index')

data.columns = columns[1:]
data.head()
data.describe()
data = data.astype(float)
data['age'] = data['age'] // 365

data.head()
data['cardio'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15, 8))

sns.countplot(x='age', hue='cardio', data=data);



plt.ylabel('count', size=15)

plt.xlabel('age', size=15)



L = plt.legend(loc='upper left', prop={'size':13})

L.get_texts()[0].set_text('No disease')

L.get_texts()[1].set_text('Disease present')
p_disease = data[data['cardio'] == 1]

p_healthy = data[data['cardio'] == 0]



plt.figure()

sns.countplot(x ='cholesterol', hue='gender', data=p_disease)


plt.figure(figsize=(8, 8))

plt.title('Glucose Levels vs Age')



sns.set(style="dark", palette="husl", color_codes=True)



sns.boxplot(x="cardio", y="age", hue="gluc", data=data)



L = plt.legend(loc='lower right', prop={'size':13})



L.get_texts()[0].set_text('Normal Glucose')

L.get_texts()[1].set_text('Above Normal Glucose')

L.get_texts()[2].set_text('High Glucose')



plt.ylabel('age', size=15)

plt.xlabel('Cardiovascular Disease Present or Not', size=15)



plt.xticks([0, 1], ['Healthy', 'Disease'])

plt.show()
X = data.drop('cardio', axis=1)

y = data['cardio']
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation
model = Sequential()

model.add(Dense(64, activation='relu', input_shape = (11,) ))

model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split = .2)