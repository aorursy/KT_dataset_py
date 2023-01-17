%matplotlib inline

%load_ext autoreload

%autoreload 2

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 
data_path = '../input/Admission_Predict_Ver1.1.csv'



admissions = pd.read_csv(data_path)
admissions.head()
admissions.describe()
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(admissions.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()

cols=admissions.drop(labels='Serial No.',axis=1)

sns.pairplot(data=cols,hue='Research')
y = np.array([admissions["TOEFL Score"].min(),admissions["TOEFL Score"].mean(),admissions["TOEFL Score"].max()])

x = ["Lowest","Mean","Highest"]

plt.bar(x,y)

plt.title("TOEFL Scores")

plt.xlabel("Level")

plt.ylabel("TOEFL Score")

plt.show()
admissions["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))

plt.title("GRE Scores")

plt.xlabel("GRE Score")

plt.ylabel("Frequency")

plt.show()
admissions.plot(kind='scatter', x='University Rating', y='CGPA')
s = admissions[admissions["Chance of Admit "] >= 0.75]["University Rating"].value_counts()

plt.title("University Ratings of Candidates with a 75% acceptance chance")

s.plot(kind='bar',figsize=(20, 10))

plt.xlabel("University Rating")

plt.ylabel("Candidates")

plt.show()
fig = plt.figure(figsize = (20, 25))

j = 0

for i in admissions.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(admissions[i][admissions['Chance of Admit ']<0.72], color='r', label = 'Not Got Admission')

    sns.distplot(admissions[i][admissions['Chance of Admit ']>0.72], color='g', label = 'Got Admission')

    plt.legend(loc='best')

fig.suptitle('Admission Chance In University ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
dummy_fields = ['University Rating', 'Research']

one_hot_admissions = admissions[:]

for each in dummy_fields:

    dummies = pd.get_dummies(one_hot_admissions[each], prefix=each, drop_first=False)

    one_hot_admissions = pd.concat([one_hot_admissions, dummies], axis=1)



to_be_dropped = ['University Rating', 'Research', 'Serial No.']

one_hot_admissions = one_hot_admissions.drop(to_be_dropped, axis=1)

one_hot_admissions.head()


processed_data = one_hot_admissions[:]



processed_data = processed_data/processed_data.max()

#processed_data = (processed_data - np.min(processed_data)) / (np.max(processed_data) - np.min(processed_data))

train_features = processed_data.drop('Chance of Admit ', axis=1)

train_targets = processed_data['Chance of Admit '].values



###This is another option####

from sklearn.model_selection import train_test_split

train_features,test_features,train_targets,test_targets = train_test_split(train_features,train_targets,test_size = 0.20,random_state = 42)



# Imports

import numpy as np

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



# Building the model

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(train_features.shape[1],)))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compiling the model

model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

model.summary()
# Training the model

history = model.fit(train_features, train_targets, validation_split=0.2, epochs=100, batch_size=8, verbose=0)
#print(vars(history))

plt.plot(history.history['loss'])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Evaluating the model on the training and testing set

score = model.evaluate(train_features, train_targets)

print("score: ", score)

print("\n Training Accuracy:", score)

score = model.evaluate(test_features, test_targets)

print("score: ", score)

print("\n Testing Accuracy:", score)
y_pred = model.predict(test_features)

plt.plot(test_targets)

plt.plot(y_pred)

plt.title('Prediction')