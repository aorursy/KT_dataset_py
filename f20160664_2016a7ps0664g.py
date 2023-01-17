import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch as torch

%matplotlib inline

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=",")

dft = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=",")
df.head()
df.info()
df.Size.unique()
df.drop_duplicates(inplace = True)

dft.drop_duplicates(inplace = True)
dft.head()
df = pd.get_dummies(df, columns=['Size'], prefix = ['S'])



df.head()
dft = pd.get_dummies(dft, columns=['Size'], prefix = ['S'])
df = pd.get_dummies(df, columns=['Number of Quantities'], prefix = ['NQ'])

dft = pd.get_dummies(dft, columns=['Number of Quantities'], prefix = ['NQ'])

df.head()
df = pd.get_dummies(df, columns=['Number of Insignificant Quantities'], prefix = ['NIQ'])

dft = pd.get_dummies(dft, columns=['Number of Insignificant Quantities'], prefix = ['NIQ'])

df.head()
df = pd.get_dummies(df, columns=['Number of Special Characters'], prefix = ['NSC'])

dft = pd.get_dummies(dft, columns=['Number of Special Characters'], prefix = ['NSC'])

df.head()
dft.info()
df["S_?"].value_counts()
df.info()
df = df.drop('S_?', axis=1)

df = df.drop('NSC_?', axis=1)

df = df.drop('NQ_?', axis=1)

df = df.drop('NIQ_?', axis=1)
df.info()
df = df[df["Difficulty"] != '?']

df.head()
df = df[df["Total Number of Words"] != '?']

df.head()
df.info()
df = df.astype(str).astype(float)
df.info()
corr = df.corr()



fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True)
df = df.drop(['ID'], axis = 1)
dft = dft.drop(['ID'], axis = 1)
dft.head()
y = df['Class']

X = df.drop(['Class'], axis = 1)

X.head()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 13)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 13)
X_train = X_train.reset_index().drop(['index'], axis = 1)

X_val = X_val.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)
numerical_columns = ["Total Number of Words","Total Number of Characters","Number of Sentences","First Index","Second Index","Difficulty", "Score"]

numerical_df_train = pd.DataFrame(X_train[numerical_columns])

rest_df_train = X_train.drop(numerical_columns, axis = 1)

numerical_df_val = pd.DataFrame(X_val[numerical_columns])

rest_df_val = X_val.drop(numerical_columns, axis = 1)

numerical_df_test = pd.DataFrame(X_test[numerical_columns])

rest_df_test = X_test.drop(numerical_columns, axis = 1)
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaler = StandardScaler()

numerical_df_train_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_train),columns=numerical_columns)



scaler = StandardScaler()

numerical_df_val_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_val),columns=numerical_columns)



scaler = StandardScaler()

numerical_df_test_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_test),columns=numerical_columns)
X_train = pd.concat([numerical_df_train_scaled, rest_df_train], axis=1)

X_val = pd.concat([numerical_df_val_scaled, rest_df_val], axis=1)

X_test = pd.concat([numerical_df_test_scaled, rest_df_test], axis=1)
model = Sequential()

model.add(Dense(50, input_dim=21, activation='relu'))

#model.add(Dropout(rate=0.3))

#model.add(Dense(128, activation='relu'))

#model.add(Dropout(rate=0.3))

#model.add(Dropout(rate=0.3))

#model.add(Dense(128, activation='relu'))

#model.add(Dropout(rate=0.3))

model.add(Dense(100, activation='relu'))

#model.add(Dropout(rate=0.3))

model.add(Dense(6, activation='softmax'))

# Compile model

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs = 25, batch_size = 4)
df.info()
NSC7 = dft["NSC_7"]

dft = dft.drop("NSC_7", axis = 1)
l = ['NSC_5', 'NSC_6']

for cols in l:

    dft[cols] = 0
dft["NSC_7"] = NSC7

dft = dft.astype(float)
predictions = model.predict(dft)
len(predictions)
labels = np.argmax(predictions, axis=-1)    

print(labels)

array = np.arange(371,530)
dataset = pd.DataFrame({'ID': array[:], 'Class': labels[:]})

dataset['Class'].unique()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(dataset)