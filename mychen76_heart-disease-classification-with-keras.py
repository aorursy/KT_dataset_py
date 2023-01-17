# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import warnings

warnings.filterwarnings('ignore')
# Import libraries for data wrangling, preprocessing and visualization

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline

# Importing libraries for building the neural network

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

seed = 8

np.random.seed(seed)
# Input data files are available in the "../input/" directory.

# Read data file

data = pd.read_csv("../input/heart.csv", header=0)

# Take a look at the data

data.head(10)
# Take a end of file

data.tail(10)
print('Number of rows in the dataset: ',data.shape[0])

print('Number of columns in the dataset: ',data.shape[1])
data.info()
data.describe()
# Any empty values?

data.isnull().sum()
# draw histogram of the features 

hist = data.hist(bins=10, figsize=(16,10))
plt.figure(figsize=(8,4))

sns.distplot(data['age'],kde=False,bins=10)

print ("Age max:", data['age'].max(), " min:", data['age'].min())
plt.figure(figsize=(12,8))

# Chest Pain

plt.subplot(221)

plt.title("Chest Pain types")

labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'

sizes = [len(data[data['cp'] == 0]),len(data[data['cp'] == 1]),

         len(data[data['cp'] == 2]),len(data[data['cp'] == 3])]

plt.pie(sizes, explode=(0, 0,0,0), labels=labels,autopct='%1.1f%%', shadow=True, startangle=180)

# blood sugar

plt.subplot(222)    

plt.title("Blood sugar")

labels = 'fasting blood sugar < 120 mg/dl','fasting blood sugar > 120 mg/dl'

sizes = [len(data[data['fbs'] == 0]),len(data[data['cp'] == 1])]

plt.pie(sizes, explode=(0.1, 0), labels=labels, autopct='%1.1f%%', shadow=True, startangle=180)
# draw a heatmap

sns.set_style('whitegrid')

plt.figure(figsize=(15,8))

sns.heatmap(data.corr(), annot = True, linewidths=.2)

plt.show()
rcParams['figure.figsize'] = 5,3

plt.bar(data['target'].unique(), data['target'].value_counts())

plt.xticks([0, 1])

plt.xlabel('Target')

plt.ylabel('Count')

plt.title('Count of each Target Class')
plt.figure(figsize=(15,6))

sns.countplot(x='age',data = data, hue = 'target')

plt.show()
# Select the columns to use for prediction in the neural network

X= data.drop('target',axis=1)

Y=data['target']

print (X.shape, Y.shape, data.columns)
# split data into train, test

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=39, shuffle=True)

#kipping y since value already 1 or 0

# encoder = LabelEncoder()

# encoder.fit(Y)

# encoded_Y = encoder.transform(Y)



# normalize data

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)

X_test = pd.DataFrame(X_test_scaled)



print (X_train.shape, y_train.shape)

print (X_train.shape, y_test.shape)

print (data.columns)
#let's build a xgboot classifier to find out feature importance

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

best_xgc_score=0

model = XGBClassifier(max_depth=7)

model.fit(X_train,y_train,eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False)

predictions = model.predict(X_test)

best_xgc_score = accuracy_score(y_test,predictions)

print ("XGBClassifier accuracy: ", best_xgc_score)
print (X.columns) 

print (model.feature_importances_*100)

print (model.classes_)  # output

# visualize it

plt.figure(figsize=(16,5))

rf_scores=model.feature_importances_*100

plt.bar([i for i in range(len(X.columns))], rf_scores, width = 0.8)

for i in range(len(X.columns)):

    plt.text(i, rf_scores[i], rf_scores[i])

plt.xlabel('Feature')

plt.ylabel('Scores')

plt.title('Feature importances')
# Define some useful callbacks

#Reduce learning rate when a metric has stopped improving.

reducelrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

# Stop training when a monitored quantity has stopped improving. 

# By default, mode is set to ‘auto‘ and knows that you want to minimize loss or maximize accuracy.

early_stopping_monitor=EarlyStopping(monitor='val_loss',verbose=1, patience=30, baseline=0.4, )

# Save the model after every epoch.

best_trained_model_file= 'best_trained_model.h5'

checkpoint = ModelCheckpoint(best_trained_model_file, verbose=0, monitor='val_loss',save_best_only=True, mode='auto')  

#place callbacks want to enable on this list

callbacks=[checkpoint, reducelrp]
# create model with fully connected layers with dropout regulation

model = Sequential()

model.add(Dense(12, input_dim=13, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(6, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="Adamax", metrics=['accuracy'])

model.summary()
%%time

# fit the model

print ("trainning model....  please wait!")

history=model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=6, callbacks=callbacks,verbose=0)

plt.plot(history.history['acc'])

plt.show()

print ("model training - finished")
print("Evaluate model against trained data")

score = model.evaluate(X_train, y_train, verbose=0)

print("score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))



print("Evaluate model against new data")

score = model.evaluate(X_test, y_test, verbose=0)

print("score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print("Model prediction test")

# prediction return class type (1 or 0)

y_pred_class = model.predict_classes(X_test)

# prediction return proability percentage

y_pred_prob = model.predict(X_test)



print ("#  original | predicted  | probability  ")

for idx, label in enumerate(y_test):

    print ("%s     | %s  | %s |   %.2f%%" % (str(idx), str(label), str(y_pred_class[idx]), float(y_pred_prob[idx])*100))



# manually calculate accuracy rate

print("")

count = len(["ok" for idx, label in enumerate(y_test) if label == y_pred_class[idx]])

print ("Manually calculated accuracy is: %.2f%%" % ((float(count) / len(y_test))*100))

# using accuracy_score()

print ("Keras accuracy_score() is: %.2f%%" %  (accuracy_score(y_test, y_pred_class)*100))

print("")

print ("Simple confusion matrix ")

cm = confusion_matrix(y_test,y_pred_class)

print (cm)
%%time

# define 10-fold cross validation test harness

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []

print ("running model performance validation... please wait!")

for train, test in kfold.split(X, Y):

    # create model

    model = Sequential()

    model.add(Dense(12, input_dim=13, kernel_initializer='uniform', activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(6, kernel_initializer='uniform', activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="Adamax", metrics=['accuracy'])

    # Fit the model

    history=model.fit(X_train, y_train, epochs=100, batch_size=6, verbose=0)    

    # evaluate the model

    scores = model.evaluate(X_test, y_test, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)



print ("done.")

print ("summary report on mean and std.")

# The average and standard deviation of the model performance 

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# save trained model

#trained_model_file="trained_heart_model.h5"

#model.save_weights(trained_model_file)

#print("Saved trained model to disk as h5 file :", trained_model_file)