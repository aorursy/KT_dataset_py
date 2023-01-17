import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.isna().sum()
df["Cabin"].fillna('X', inplace=True)

df["Cabin"] = df["Cabin"].str[0]

df["Cabin"] = pd.Categorical(df["Cabin"])

hot_cabin = pd.get_dummies(df["Cabin"], prefix="cabin")

del df["Cabin"]

df = pd.concat([df, hot_cabin], axis=1)
df["Pclass"] = pd.Categorical(df["Pclass"])

hot_pclass = pd.get_dummies(df["Pclass"], prefix="class")

del df["Pclass"]

df = pd.concat([df, hot_pclass], axis=1)
df["Sex"] = pd.Categorical(df["Sex"])

hot_sex = pd.get_dummies(df["Sex"], prefix="sex")

del df["Sex"]

df = pd.concat([df, hot_sex], axis=1)

del df["sex_male"]
df["Embarked"] = pd.Categorical(df["Embarked"])

hot_embarked = pd.get_dummies(df["Embarked"], prefix="embarked")

del df["Embarked"]

df = pd.concat([df, hot_embarked], axis=1)
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

df.dropna(inplace=True)

df.head()
import seaborn as sb

import matplotlib.pyplot as plt



plt.figure(figsize=(16,6))

sb.heatmap(df.corr(), annot=True)
# df = df.sample(frac=1).reset_index(drop=True)



divider = int(0.7 * len(df))



train_df = df.iloc[:divider]

validation_df = df.iloc[divider:]
train_x = np.asarray(train_df[train_df.columns[1:]])

train_y = np.asarray(train_df[train_df.columns[0]])



validation_x = np.asarray(validation_df[validation_df.columns[1:]])

validation_y = np.asarray(validation_df[validation_df.columns[0]])
from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()



train_x = scaler.fit_transform(train_x)

validation_x = scaler.fit_transform(validation_x)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.regularizers import l2



model = Sequential()



model.add(Dense(32, input_dim=20, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(train_x, train_y, batch_size=1, epochs=5, validation_data=(validation_x, validation_y))



validation_loss, validation_acc = model.evaluate(validation_x,  validation_y, verbose=2)



print(f'Validation accuracy: {validation_acc:.3f}, validation loss: {validation_loss:.3f}')