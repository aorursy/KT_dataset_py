# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import math
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, ActivityRegularization,Dropout
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
# Describe data
grid_rows = 1
grid_columns = 5

survivor_df = train_df[train_df.Survived == 1]
died_df = train_df[train_df.Survived == 0]
plt.rcParams["figure.figsize"] = (30,5)
ax = plt.subplot2grid((grid_rows,grid_columns), (0,0))
ax.bar(['Survived','Died'],[survivor_df.count().PassengerId,died_df.count().PassengerId])
ax.set_title("Survivor count")

ax = plt.subplot2grid((grid_rows,grid_columns), (0,1))
ax.bar(['Male','Female'],[survivor_df[survivor_df.Sex == 'male'].count().PassengerId,survivor_df[survivor_df.Sex == 'female'].count().PassengerId])
ax.set_title("Survivor wrt sex")

no_age_df = train_df[train_df.Age.notnull()]
ax = plt.subplot2grid((grid_rows,grid_columns), (0,2))
counts, xedges, yedges, im = ax.hist2d(no_age_df.Survived.values, no_age_df.Age.values, normed=False, bins=[2,16])
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1])
ax.set_xticklabels(['Died', 'Survived'])
ax.set_title("Survivor wrt age")

class_1_count = train_df[train_df.Pclass == 1].count().PassengerId
class_2_count = train_df[train_df.Pclass == 2].count().PassengerId
class_3_count = train_df[train_df.Pclass == 3].count().PassengerId
ax = plt.subplot2grid((grid_rows,grid_columns), (0,3))
ax.bar(['1st Class','2nd Class', '3rd Class'],[class_1_count,class_2_count, class_3_count])
ax.set_title("Passenger count wrt class")

class_1_survived = train_df[(train_df.Pclass == 1) & (train_df.Survived == 1)].count().PassengerId
class_2_survived = train_df[(train_df.Pclass == 2) & (train_df.Survived == 1)].count().PassengerId
class_3_survived = train_df[(train_df.Pclass == 3) & (train_df.Survived == 1)].count().PassengerId

ax = plt.subplot2grid((grid_rows,grid_columns), (0,4))
ax.bar(['1st Class','2nd Class', '3rd Class'],[class_1_survived/class_1_count,class_2_survived/class_2_count,class_3_survived/class_3_count])
ax.set_title("Survivor % wrt Passenger tickets")



plt.show()



# Print correlation
corr = train_df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, annot=True, linewidth=0.5)
pass
# Age prediction model
def age_row_formatter(row):
    sex = 1 if row['Sex'] == 'male' else 0
    sib = row['SibSp']
    parch = row['Parch']
    class_1 = 1 if row['Pclass'] == 1 else 0
    class_2 = 1 if row['Pclass'] == 2 else 0
    class_3 = 1 if row['Pclass'] == 3 else 0
    depart_c = 1 if row['Embarked'] == 'C' else 0
    depart_q = 1 if row['Embarked'] == 'Q' else 0
    depart_s = 1 if row['Embarked'] == 'S' else 0

    # normalize
    sib = sib / 5
    parch = parch / 5
    return [sex, sib, parch, class_1, class_2, class_3, depart_c, depart_q, depart_s]
    
def data_age_formatter(df):
    data_x = []
    data_y = []
    mean_age = df.Age.mean()
    for index, row in df.iterrows():
        age = mean_age if pd.isna(row['Age']) else row['Age']
        data_x.append(age_row_formatter(row))
        data_y.append(age)
    return np.array(data_x), np.array(data_y)

age_x, age_y = data_age_formatter(train_df.dropna())
age_model = Sequential()
age_model.add(Dense(128, input_shape=(age_x.shape[1],), activation='relu'))
age_model.add(Dropout(0.2))
age_model.add(Dense(1, activation='relu'))
age_model.compile('rmsprop', loss='mean_squared_error', metrics=['accuracy'])
history = age_model.fit(age_x,age_y, epochs=100, verbose=0, validation_split=0.1)
print(history.history['val_acc'][-1])
# print training
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Survival prediction model

def data_formatter(df):
    data_x = []
    data_y = []
    mean_age = df.Age.mean()
    for index, row in df.iterrows():
        if 'Survived' not in row.keys():
            y = None
        else:
            y = np.array([0,1]) if row['Survived'] == 1 else np.array([1,0])
        sex = 1 if row['Sex'] == 'male' else 0
        if pd.isna(row['Age']):
            age_input = np.array([age_row_formatter(row)])
            age = age_model.predict(age_input)[0]
        else:
            age = row['Age']
        sib = row['SibSp']
        parch = row['Parch']
        pclasss = row['Pclass']
        fare = row['Fare']
        depart_c = 1 if row['Embarked'] == 'C' else 0
        depart_q = 1 if row['Embarked'] == 'Q' else 0
        depart_s = 1 if row['Embarked'] == 'S' else 0
        
        # normalize
        age = age / 80
        sib = sib / 10
        parch = parch / 10
        pclasss = pclasss / 3
        
        data_x.append([sex, age, sib, parch, pclasss, depart_c, depart_q, depart_s, fare])
        data_y.append(y)
    return np.array(data_x), np.array(data_y)

x, y = data_formatter(train_df.dropna())
model = Sequential()
model.add(Dense(1024, input_shape=(x.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))
#model.add(ActivityRegularization(l1=0.01, l2=0.01))
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x,y, epochs=200, verbose=0, validation_data=(x,y))

# print training
print('Accuracy without missing age: {}'.format(history.history['val_acc'][-1]))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
x,y = data_formatter(test_df)
predictions = model.predict(x)
predictions = np.resize(predictions, (x.shape[0],2))

predictions = [np.argmax(p) for p in predictions]
test_df['Survived'] = pd.Series(predictions)
test_df[['PassengerId','Survived']].to_csv('submission.csv', index=False)