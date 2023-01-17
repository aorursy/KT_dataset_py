import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sns
import tensorflow as tf
# Reading data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
print('Train data size',train.shape)
print('Test data size',test.shape)

print(train['Survived'].value_counts(),'\n')
print(Counter(train['Sex']),'\n')
print(train['SibSp'].value_counts(),'\n')
print(train['Parch'].value_counts(),'\n')
print(train['Embarked'].value_counts(),'\n')
rcParams['figure.figsize'] = 10,5
sb.barplot(x = train['Survived'].value_counts().index, y = train['Survived'].value_counts().values)
plt.title('Survival counts')
plt.xlabel('Survived')
plt.ylabel('No of passengers')
plt.show()
gender = pd.crosstab(train['Survived'],train['Sex'])
gender.plot(kind="bar",title='No of passengers survived')
plt.show()
rcParams['figure.figsize'] = 10,5
ax = train['Age'].hist(bins = 15,alpha = 0.9, color = 'green')
ax.set(xlabel = 'Age',ylabel = 'Count',title = 'Visualization of Ages')
plt.show()
rcParams['figure.figsize'] = 10,10
sb.heatmap(train.corr(),annot = True,square = True,linewidths = 2,linecolor = 'black')
y = train["Survived"]

# getting dummy variables column

enc = LabelEncoder()

train['Sex'] = enc.fit_transform(train['Sex'])
test['Sex'] = enc.fit_transform(test['Sex'])

train['Name'] = enc.fit_transform(train['Name'])
test['Name'] = enc.fit_transform(test['Name'])

train['Cabin'] = enc.fit_transform(train['Cabin'].astype('str'))
test['Cabin'] = enc.fit_transform(test['Cabin'].astype('str'))

train['Embarked'] = enc.fit_transform(train['Embarked'].astype('str'))
test['Embarked'] = enc.fit_transform(test['Embarked'].astype('str'))

train['Ticket'] = enc.fit_transform(train['Ticket'].astype('category'))
test['Ticket'] = enc.fit_transform(test['Ticket'].astype('category'))
 
X = train
test_X = test
X.set_index(['PassengerId'],inplace = True)
test_X.set_index(['PassengerId'],inplace = True)
X = X.drop(['Survived'], axis=1)
X.tail()
test_X.head()
print(X.isnull().sum(),'\n')
print(test_X.isnull().sum())
X.fillna(X.median(), inplace=True)
test_X.fillna(test_X.mean(), inplace=True)
print(X.isnull().sum(),'\n')
print(test_X.isnull().sum())
#Normalizing

from sklearn.preprocessing import normalize

X = normalize(X)
test_X = normalize(test_X)
print("Logistic Regression:", cross_val_score(LogisticRegression(), X, y).mean())

print("SVC:", cross_val_score(SVC(), X, y).mean())

print("Random Forest:", cross_val_score(RandomForestClassifier(), X, y).mean())

print("GaussianNB:", cross_val_score(GaussianNB(), X, y).mean())

print("Decision Tree:", cross_val_score(DecisionTreeClassifier(), X, y).mean())

print("KNeighbors:", cross_val_score(KNeighborsClassifier(), X, y).mean())

print("MLP:", cross_val_score(MLPClassifier(), X, y).mean())

print("XGB-TREE:", cross_val_score(XGBClassifier(booster='gbtree'), X, y).mean())

print("XGB-DART:", cross_val_score(XGBClassifier(booster='dart'), X, y).mean())
# fit the model on the whole dataset
model_RF = RandomForestClassifier()

model_RF.fit(X, y)
y_pred = model_RF.predict(test_X)
test_X
# #K Fold Cross Validation

# from sklearn.model_selection import KFold


# kf = KFold(n_splits=5, random_state=42, shuffle=True)

# for train_index, val_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", val_index)
#     X_train, X_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]
# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)

# #reshape for rnn

# X_train = X_train.reshape(-1, 1, 10)
# X_val  = X_val.reshape(-1, 1, 10)
# y_train = y_train.values #convert pd to array
# y_train = y_train.reshape(-1, 1,)
# y_val = y_val.values #convert pd to array
# y_val = y_val.reshape(-1, 1,)

# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# from tensorflow.keras.layers import Conv2D,LSTM,LeakyReLU, MaxPooling2D,Concatenate,Input, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.models import Model


# # create model
    

# #input 
# input_layer = Input(shape=(1,10))
# main_rnn_layer = LSTM(1024, return_sequences=True, recurrent_dropout=0.9)(input_layer)

    
# #output
# rnn = LSTM(1024)(main_rnn_layer)
# dense = Dense(512)(rnn)
# dropout_c = Dropout(0.8)(dense)
# dense = Dense(265)(dropout_c)
# dropout_c = Dropout(0.5)(dense)
# dense = Dense(128)(dropout_c)
# dropout_c = Dropout(0.3)(dense)

# classes = Dense(1, activation= LeakyReLU(alpha=0.1),name="class")(dropout_c)

# model = Model(input_layer, classes)

# # Compile model
# callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
#              EarlyStopping(monitor='val_loss', patience=20),
#              ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
# model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")

# model.summary()
# # Fit the model
# history = model.fit(X_train, y_train, 
#           epochs = 1000, 
#           batch_size = 128, 
#           validation_data=(X_val,  y_val), 
#           callbacks=callbacks)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Loss over epochs')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='best')
# plt.show()
# test_X.shape

# test_X = test_X.reshape(-1, 1,10)
# test_predict = np.where(test_predict > 0.5, 1, 0)
test_predict = pd.Series(y_pred)
test.reset_index(inplace = True)
test.head()
predict = test['PassengerId']
predict = pd.concat([predict,test_predict], axis=1)
predict.rename(columns={0: "Survived"},inplace=True)
predict.to_csv("submission.csv",index=False)
predict
