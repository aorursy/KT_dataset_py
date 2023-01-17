%matplotlib inline

%load_ext autoreload

%autoreload 2

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

from math import pi
data_path = '../input/heart.csv'



df = pd.read_csv(data_path)
df.head()
df.describe()
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
sns.pairplot(data=df)
#print("Men vs Women Count\n", df.sex.value_counts())

men_count = len(df[df['sex']== 1])

women_count = len(df[df['sex']==0])



plt.figure(figsize=(8,6))



# Data to plot

labels = 'Men','Women'

sizes = [men_count,women_count]

colors = ['skyblue', 'yellowgreen']

explode = (0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.2f%%', shadow=True)

plt.show()

 
sns.countplot(x='target', data=df)
print("People having heart diseace vs people who doesn't: \n", df.target.value_counts())

heart_disease = len(df[df['target']==1])

no_heart_disease = len(df[df['target']==0])

labels = ["Heart Diesease", "NO Heart Disease"]

sizes = [heart_disease, no_heart_disease]

colors = ['skyblue', 'yellowgreen']

plt.figure(figsize=(8,6))



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.2f%%', shadow=True)

plt.show()

 
pd.crosstab(df.age, df.target).plot(kind='bar', figsize=(20, 10))

plt.title("People having heart disease vs people not having heart disease for a given age")

plt.xlabel("Age Distribution")

plt.ylabel("Heart Disease Frequency")



my_colors = 'yr'

pd.crosstab(df.thalach, df.target).plot(kind='bar', figsize=(20,10), color=my_colors)

plt.title("Heart diseases frequency for Maximum heart rate")

plt.xlabel("Maximum Heart Rate for a person")

plt.ylabel("Heart Disease Frequency")
pd.crosstab(df.cp, df.target).plot(kind='bar', figsize = (20,10))

plt.title("Heart disease frequency over Chesr Pain Type")

plt.xlabel("Cheast Pain Type ")

plt.ylabel("Heart Disease Frequency")
pd.crosstab(df.slope, df.target).plot(kind='bar', figsize=(10,10))

plt.title("Heart disease frequency over Slope")

plt.xlabel("Slope Frequency")

plt.ylabel("Heart Disease Frequency")


a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)



to_be_dropped = ['cp', 'thal', 'slope']

df = df.drop(to_be_dropped, axis=1)

df.head()
df = (df - np.min(df)) / (np.max(df) - np.min(df)).values
features = df.drop("target", axis=1)

targets = df.target.values
from sklearn.model_selection import train_test_split

train_features,test_features,train_targets,test_targets = train_test_split(features,targets,test_size = 0.20,random_state = 42)



# Imports

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



# Building the model

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(train_features.shape[1],)))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compiling the model

model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

model.summary()
history = model.fit(train_features, train_targets, validation_split=0.2, epochs=100, batch_size=16, verbose=1)
#print(vars(history))

plt.plot(history.history['loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(test_features)

plt.plot(test_targets)

plt.plot(y_pred)

plt.title('Prediction')