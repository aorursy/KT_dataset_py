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
#import numpy as np

#import pandas as pd

import math

import seaborn as sns

import re



from sklearn.model_selection import KFold, train_test_split

from sklearn.preprocessing import StandardScaler



import keras

from keras.engine.input_layer import Input

from keras.layers import Dense, Activation, Dropout, Add, PReLU, LeakyReLU, Reshape, Conv2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l1, l2, l1_l2

from keras.models import Model

from keras.callbacks import EarlyStopping
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

df_train.head()
sns.heatmap(pd.get_dummies(df_train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]], columns = ["Pclass", "Sex", "Embarked"]).corr())

print(pd.get_dummies(df_train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]], columns = ["Pclass", "Sex", "Embarked"]).corr()["Survived"])
df_train = df_train.drop("Name", axis = 1)

#df_train = df_train.drop("Ticket", axis = 1)

df_train = df_train.drop("Cabin", axis = 1)



df_test = df_test.drop("Name", axis = 1)

#df_test = df_test.drop("Ticket", axis = 1)

df_test = df_test.drop("Cabin", axis = 1)
df_train.plot.hist(y = ['Fare'], bins = 100)
df_train_pp = pd.get_dummies(df_train, columns = ["Survived", "Pclass", "Sex", "Embarked"])

df_test_pp = pd.get_dummies(df_test, columns = ["Pclass", "Sex", "Embarked"])

#pp means preprocessed
'''

def convert_Fare(fare):

    fare0 = fare

    preprocessed = 0

    if fare > 100:

        fare0 = 100

    if fare0 != 0:

        fare_log = np.log(fare0)

        preprocessed = fare_log / np.log(100)

    return preprocessed



df_train_pp["Fare"] = df_train_pp["Fare"].map(convert_Fare)

df_test_pp["Fare"] = df_test_pp["Fare"].map(convert_Fare)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train_pp["Fare"].values, axis = 1))

df_train_pp["Fare"] = pd.DataFrame(scaler.transform(np.expand_dims(df_train_pp["Fare"].values, axis = 1)), columns = ["Fare"])

df_test_pp["Fare"] = pd.DataFrame(scaler.transform(np.expand_dims(df_test_pp["Fare"].values, axis = 1)), columns = ["Fare"])
df_train_pp.head()
df_train_pp.describe()
def convert_Age(age):

    preprocessed = age

    if np.isnan(age):

        preprocessed = np.nan

    elif age < 1:

        preprocessed = np.nan

    return preprocessed



df_train_pp["Age"] = df_train_pp["Age"].map(convert_Age)

df_test_pp["Age"] = df_test_pp["Age"].map(convert_Age)



df_train_pp["Age"] = df_train_pp["Age"].fillna(df_train_pp["Age"].mean())

df_test_pp["Age"] = df_test_pp["Age"].fillna(df_test_pp["Age"].mean())
'''

scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train_pp["Age"].fillna(df_train_pp["Age"].mean()).values, axis = 1))

df_train_pp["Age"] = pd.DataFrame(scaler.transform(np.expand_dims(df_train_pp["Age"].fillna(df_train_pp["Age"].mean()).values, axis = 1)), columns = ["Age"])

df_test_pp["Age"] = pd.DataFrame(scaler.transform(np.expand_dims(df_test_pp["Age"].fillna(df_train_pp["Age"].mean()).values, axis = 1)), columns = ["Age"])

'''
df_train_pp["Age"] = pd.qcut(df_train_pp["Age"], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

df_test_pp["Age"] = pd.qcut(df_test_pp["Age"], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
df_train_pp["SibSp"] = pd.qcut(df_train_pp["SibSp"], 5, labels=['Q1', 'Q2'], duplicates='drop')

df_test_pp["SibSp"] = pd.qcut(df_test_pp["SibSp"], 5, labels=['Q1', 'Q2'], duplicates='drop')
df_train_pp["Parch"] = pd.qcut(df_train_pp["Parch"], 5, labels=['Q1', 'Q2'], duplicates='drop')

df_test_pp["Parch"] = pd.qcut(df_test_pp["Parch"], 5, labels=['Q1', 'Q2'], duplicates='drop')
'''

scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train_pp["SibSp"].values, axis = 1))

df_train_pp["SibSp"] = pd.DataFrame(scaler.transform(np.expand_dims(df_train_pp["SibSp"].values, axis = 1)), columns = ["SibSp"])

df_test_pp["SibSp"] = pd.DataFrame(scaler.transform(np.expand_dims(df_test_pp["SibSp"].values, axis = 1)), columns = ["SibSp"])

'''
'''

scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train_pp["Parch"].values, axis = 1))

df_train_pp["Parch"] = pd.DataFrame(scaler.transform(np.expand_dims(df_train_pp["Parch"].values, axis = 1)), columns = ["Parch"])

df_test_pp["Parch"] = pd.DataFrame(scaler.transform(np.expand_dims(df_test_pp["Parch"].values, axis = 1)), columns = ["Parch"])

'''
df_train_pp = pd.get_dummies(df_train_pp, columns = ["Age", "SibSp", "Parch"])

df_test_pp = pd.get_dummies(df_test_pp, columns = ["Age", "SibSp", "Parch"])
'''

def convert_Age(age):

    preprocessed = 0

    if np.isnan(age) == False:

        #preprocessed = age / df_train_pp["Age"].max(axis = 1)

        preprocessed = age / 80

    return preprocessed



df_train_pp["Age"] = df_train_pp["Age"].map(convert_Age)

df_test_pp["Age"] = df_test_pp["Age"].map(convert_Age)





def convert_SibSp(sibsp):

    preprocessed = 0

    if np.isnan(sibsp) == False:

        preprocessed = sibsp / 8

    return preprocessed



df_train_pp["SibSp"] = df_train_pp["SibSp"].map(convert_SibSp)

df_test_pp["SibSp"] = df_test_pp["SibSp"].map(convert_SibSp)





def convert_Parch(parch):

    preprocessed = 0

    if np.isnan(parch) == False:

        preprocessed = parch / 6

    return preprocessed



df_train_pp["Parch"] = df_train_pp["Parch"].map(convert_Parch)

df_test_pp["Parch"] = df_test_pp["Parch"].map(convert_Parch)

'''
def convert_Ticket(ticket):

    str = re.sub("\\D", "", ticket)

    if str != '':

        preprocessed = int(str)

    else:

        preprocessed = 0

    

    if preprocessed > 1000000:

        preprocessed = preprocessed / 1000

    

    return preprocessed



df_train_pp["Ticket"] = df_train_pp["Ticket"].map(convert_Ticket)

df_test_pp["Ticket"] = df_test_pp["Ticket"].map(convert_Ticket)
df_train_pp.plot.hist(y = ['Ticket'], bins = 100)
df_train_pp.describe()
df_train_pp["Ticket"] = pd.qcut(df_train_pp["Ticket"], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

df_test_pp["Ticket"] = pd.qcut(df_test_pp["Ticket"], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

df_train_pp = pd.get_dummies(df_train_pp, columns = ["Ticket"])

df_test_pp = pd.get_dummies(df_test_pp, columns = ["Ticket"])
'''

scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train_pp["Ticket"].values, axis = 1))

df_train_pp["Ticket"] = pd.DataFrame(scaler.transform(np.expand_dims(df_train_pp["Ticket"].values, axis = 1)), columns = ["Ticket"])

df_test_pp["Ticket"] = pd.DataFrame(scaler.transform(np.expand_dims(df_test_pp["Ticket"].values, axis = 1)), columns = ["Ticket"])

'''
df_train_pp.head()
na_train_x = df_train_pp[["Age_Q1", "Age_Q2", "Age_Q3", "Age_Q4", "Age_Q5", "SibSp_Q1", "SibSp_Q2", "Parch_Q1", "Parch_Q2", "Ticket_Q1", "Ticket_Q2", "Ticket_Q3", "Ticket_Q4", "Ticket_Q5", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_S"]].values

na_train_y = df_train_pp[["Survived_0", "Survived_1"]].values



na_test_x = df_test_pp[["Age_Q1", "Age_Q2", "Age_Q3", "Age_Q4", "Age_Q5", "SibSp_Q1", "SibSp_Q2", "Parch_Q1", "Parch_Q2", "Ticket_Q1", "Ticket_Q2", "Ticket_Q3", "Ticket_Q4", "Ticket_Q5", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_S"]].values

na_test_id = df_test_pp[["PassengerId"]].values
df_train_pp.head(10)
'''def build_model():

    input_layer = Input((12,))

    #x = BatchNormalization()(input_layer)

    x = Dense(units = 64, activity_regularizer=l1_l2())(input_layer)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Add()([x, input_layer])

    

    x = BatchNormalization()(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Add()([x, input_layer])

    

    x = BatchNormalization()(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Add()([x, input_layer])

    

    x = BatchNormalization()(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Dense(units = 12)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Add()([x, input_layer])

    

    x = BatchNormalization()(x)

    x = Dense(units = 6)(x)

    x = Activation('relu')(x)

    x = Dense(units = 4)(x)

    x = Activation('relu')(x)

    x = Dense(units = 2)(x)

    x = Activation('softmax')(x)

    

    model = Model(inputs = input_layer, outputs = x)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    return model'''
def build_model():

    input_layer = Input((22,))

    #x = BatchNormalization()(input_layer)

    x = Dense(units = 144)(input_layer)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    

    x = Reshape((12,12,1))(x)

    

    x = Conv2D(filters = 16, kernel_size = (3, 3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(filters = 32, kernel_size = (3, 3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(filters = 64, kernel_size = (3, 3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    

    x = GlobalAveragePooling2D()(x)

    

    x = Dense(units = 16)(x)

    x = LeakyReLU()(x)

    x = Dropout(0.25)(x)

    

    x = Dense(units = 8)(x)

    x = LeakyReLU()(x)

    

    x = Dense(units = 4)(x)

    x = LeakyReLU()(x)

    

    x = Dense(units = 2)(x)

    

    x = Activation('softmax')(x)

    

    model = Model(inputs = input_layer, outputs = x)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

    return model
X_train, X_text, Y_train, Y_test = train_test_split(na_train_x, na_train_y, test_size = 0.1)
kf = KFold(n_splits = 10, shuffle = True)

all_loss = []

all_val_loss = []

all_acc = []

all_val_acc = []

epochs = 300
for train_index, val_index in kf.split(X_train, Y_train):

    train_data = X_train[train_index]

    train_label = Y_train[train_index]

    val_data = X_train[val_index]

    val_label = Y_train[val_index]

    

    model = build_model()

    earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=5, verbose=1, mode='auto')

    history = model.fit(x = train_data, y = train_label, epochs = epochs, batch_size = 100, validation_data = (val_data, val_label), callbacks = [earlystopping_callback])

    

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    acc = history.history["acc"]

    val_acc = history.history["val_acc"]

    

    all_loss.append(loss)

    all_val_loss.append(val_loss)

    all_acc.append(acc)

    all_val_acc.append(val_acc)



average_all_loss = np.mean([i[-1] for i in all_loss])

average_all_val_loss = np.mean([i[-1] for i in all_val_loss])

average_all_acc = np.mean([i[-1] for i in all_acc])

average_all_val_acc = np.mean([i[-1] for i in all_val_acc])



print("Loss: {}, Val_Loss: {}, Accuracy: {}, Val_Accuracy: {}".format(average_all_loss, average_all_val_loss, average_all_acc, average_all_val_acc))
model = build_model()

earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1, mode='auto')

history = model.fit(x = na_train_x, y = na_train_y, epochs = epochs, batch_size = 100, callbacks = [earlystopping_callback], validation_split=0.1)
output_raw = np.expand_dims(model.predict(na_test_x).argmax(axis = -1).astype(np.uint8), axis = -1)
na_output = np.concatenate([na_test_id, output_raw], axis = 1)
df_output = pd.DataFrame(na_output, columns = ["PassengerId", "Survived"])

df_output.head()
df_output.to_csv("predict.csv", index = False)