import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
seed = 13

np.random.seed(seed)
import os

os.listdir('/kaggle/input/bitsf312-lab1')
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=",")

df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=",")

print(df.shape)

print(df_test.shape)
df.info()
df['Class'].value_counts()
df.Class.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8))
df.head()
for key in df.keys():

 print(key, ': ', pd.unique(df[key]))
for key in df_test.keys():

 print(key, ': ', pd.unique(df_test[key]))
df.dtypes
df_test.dtypes
df = df.replace({'?': None})
df.head()
plt.scatter(df['Class'],df['Score'])
# plt.plot(df_test['Score'])

print(df_test[df_test['Score']>5000].head())
df = df.drop(['ID'], axis = 1)

df_test = df_test.drop(['ID'], axis=1)

df = df[df.Score < 50000]
null_columns = df.columns[df.isnull().any()]

null_columns
df.drop_duplicates(inplace = True)
print(df.isnull().sum())
for i in range(df.shape[0]):

    if df.iloc[i]['Number of Quantities']==None:

        print(df.iloc[i]['Number of Insignificant Quantities'])

print("-------")

for i in range(df.shape[0]):

    if df.iloc[i]['Number of Insignificant Quantities']==None:

        print(df.iloc[i]['Number of Quantities'])
for i in range(df.shape[0]):

    if df.iloc[i]['Number of Special Characters']==None:

        print(df.iloc[i]['Number of Sentences'])



print("-------")



for i in range(df.shape[0]):

    if df.iloc[i]['Number of Sentences']==None:

        print(df.iloc[i]['Number of Special Characters'])



print("-------")
df = df.drop(['Number of Special Characters'], axis = 1)

df_test = df_test.drop(['Number of Special Characters'], axis=1)
df['Number of Quantities'] = df['Number of Quantities'].replace({None: 2})
df['Number of Insignificant Quantities'] = df['Number of Insignificant Quantities'].replace({None: '0'})
print(df.isnull().sum())
df = df.drop('Number of Quantities', axis=1)

df_test = df_test.drop('Number of Quantities', axis=1)

# df.dropna(inplace=True)
print(df.info())
print(df_test.info())
for key in df.keys():

 print(key, ': ', pd.unique(df[key]))
# Modify data encoding

# using function

df['Size'] = df['Size'].map({ 'Small': 0,'Medium': 1, 'Big':2, None:1})

df_test['Size'] = df_test['Size'].map({ 'Small': 0,'Medium': 1, 'Big':2, None:1})
# df = pd.get_dummies(df, columns=['Size'],prefix= ['Size'])

# df_test = pd.get_dummies(df_test, columns=['Size'],prefix= ['Size'])
df['Size']

df.dropna(inplace=True)
print(df.isnull().sum())
df.head()
df.dtypes
df['Number of Insignificant Quantities'] = df['Number of Insignificant Quantities'].astype(int)

df['Total Number of Words'] = df['Total Number of Words'].astype(int)

df['Difficulty'] = df['Difficulty'].astype(float)

# df['Size'] = df['Size'].astype(int)
df['Chars per Class'] = df['Total Number of Characters']/ df['Total Number of Words']

df_test['Chars per Class'] = df_test['Total Number of Characters']/ df_test['Total Number of Words']

df = df.drop('Total Number of Words', axis=1)

df_test = df_test.drop('Total Number of Words', axis=1)

df['New2'] = df['Total Number of Characters'] / df['Number of Sentences']

df_test['New2'] = df_test['Total Number of Characters'] / df_test['Number of Sentences']

# df = df.drop('Number of Sentences', axis=1)

# df_test = df_test.drop('Number of Sentences', axis=1)

# df['New2'] = df['Second Index'] / df['Total Number of Characters'] 

# df['New3'] = df['First Index'] / df['Total Number of Characters'] 

# df_test['New2'] = df_test['Second Index'] / df_test['Total Number of Characters'] 

# df_test['New3'] = df_test['First Index'] / df_test['Total Number of Characters'] 

# df['FS Ratio'] = df['First Index']/ df['Second Index']

# df_test['FS Ratio'] = df_test['First Index']/ df_test['Second Index']

# df = df.drop('First Index', axis=1)

# df_test = df_test.drop('First Index', axis=1)

# df = df.drop('Second Index', axis=1)

# df_test = df_test.drop('Second Index', axis=1)

# df = df.drop('Total Number of Characters', axis=1)

# df_test = df_test.drop('Total Number of Characters', axis=1)
df.shape
import seaborn as sns



corr = df.corr()



fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr, annot=True)
# for i in range(df.shape[0]):

#     print(i, ' ', df.iloc[i]['First Index'], ' ', df.iloc[i]['Second Index'])
print(df['Class'].value_counts())



df.Class.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8))
# import matplotlib 



# x = df['Class']

# y = df['Number of Sentences']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# x = df['Class']

# y = df['New2']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# x = df['Class']

# y = df['New3']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# x = df['Class']

# y = df['First Index']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# x = df['Class']

# y = df['Total Number of Characters'] / df['Number of Sentences']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# df['New 1'] = df['First Index'] / df['Second Index']

# df['New2'] = df['Second Index'] / df['Total Number of Characters'] 

# df['New3'] = df['First Index'] / df['Total Number of Characters'] 

# df.groupby("Class").mean()[['New2', 'New3', 'Total Number of Characters']]
# for key in df.keys():

#  print(key, ': ', pd.unique(df[key]))
# import matplotlib 



# x = df['Second Index']

# y = df['First Index']

# label = df['Class']

# colors = ['red','green','blue','pink','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
# import matplotlib 



# x = df['Second Index'] / df['First Index']

# y = df['Total Number of Characters']

# label = df['Class']

# colors = ['red','green','blue','purple','black','yellow']



# fig = plt.figure(figsize=(20,8))

# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))



# cb = plt.colorbar()

# loc = np.arange(0,max(label),max(label)/float(len(colors)))

# cb.set_ticks(loc)

# cb.set_ticklabels(colors)
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['Class'], prefix = ['Class'])

df_onehot.head()
# for i in range(df.shape[0]):

#     if df.iloc[i]['Number of Quantities']==3:

#         print(df.iloc[i])

#         print("------")
df_onehot.dropna(inplace=True)

df_onehot.drop_duplicates(inplace=True)
y = df_onehot[['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']]

X = df_onehot.drop(['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'], axis = 1)

X.head()
X_test = df_test
X_test.head()
print(X.shape)

print(y.shape)

print(X_test.shape)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = seed)
X_train = X_train.reset_index().drop(['index'], axis = 1)

X_val = X_val.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)
numerical_columns = [

                     'First Index',

                     'Second Index',

                        'Number of Insignificant Quantities',

#                      'Total Number of Words',

                     'Chars per Class', 

#                      'FS Ratio',

                    'New2',

#                     'New3',

                     'Total Number of Characters',

                     'Number of Sentences',

                     'Difficulty',

                    'Size',

                     'Score']

numerical_df_train = pd.DataFrame(X_train[numerical_columns])

rest_df_train = X_train.drop(numerical_columns, axis = 1)

numerical_df_val = pd.DataFrame(X_val[numerical_columns])

rest_df_val = X_val.drop(numerical_columns, axis = 1)

numerical_df_test = pd.DataFrame(X_test[numerical_columns])

rest_df_test = X_test.drop(numerical_columns, axis = 1)



print(numerical_df_train.shape)

print(rest_df_train.shape)



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler



# scaler = StandardScaler() 

# scaler = MinMaxScaler()

scaler = RobustScaler()

numerical_df_train_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_train),columns=numerical_columns)



# scaler = StandardScaler()

numerical_df_val_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_val),columns=numerical_columns)



numerical_df_test_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_test),columns=numerical_columns)



print(numerical_df_train.shape)

print(rest_df_train.shape)



X_train = pd.concat([numerical_df_train_scaled.reset_index(drop=True), rest_df_train.reset_index(drop=True)], axis=1)

X_train.reset_index()

X_val = pd.concat([numerical_df_val_scaled.reset_index(drop=True), rest_df_val.reset_index(drop=True)], axis=1)

X_val.reset_index()

X_test = pd.concat([numerical_df_test_scaled.reset_index(drop=True), rest_df_test.reset_index(drop=True)], axis=1)

X_test.reset_index()
X_train = X_train.reset_index().drop(['index'], axis = 1)

X_val = X_val.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)
X_test.head()
X_train.head()
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras import regularizers
y_train = y_train.reset_index().drop(['index'], axis = 1)

y_val = y_val.reset_index().drop(['index'], axis = 1)
y_train.head()
y_val.head()
import keras

keras.backend.clear_session()



# import regularizer

from keras.regularizers import l1_l2



model = Sequential()

model.add(Dense(20, input_dim=10, activation='relu', kernel_regularizer=l1_l2(0.001)))

model.add(Dropout(0.2, seed=seed))

model.add(Dense(40, activation='relu', kernel_regularizer=l1_l2(0.001)))

model.add(Dropout(0.2, seed=seed))

model.add(Dense(10, activation='relu', kernel_regularizer=l1_l2(0.001)))

# model.add(Dropout(0.2, seed=seed))

# model.add(Dense(25, activation='relu', kernel_regularizer=l1_l2(0.001)))

model.add(Dropout(0.1, seed=seed))

model.add(Dense(6, activation='softmax'))
# Set callback functions to early stop training and save the best model so far

# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# callbacks = [EarlyStopping(monitor='val_accuracy', patience=5)]



# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0001)
optimizer = Adam(lr=0.001)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=2)
# plot history

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='test')

plt.legend()

plt.show()
plt.plot(history.history['loss'], label='train_loss')

plt.plot(history.history['val_loss'], label='test_loss')

plt.legend()

plt.show()
# model.save_weights("model-v5.h5")

# model.load_weights("model-v5.h5")
y_prob = model.predict(X_test) 

y_classes = y_prob.argmax(axis=-1)

print(y_classes)
df_submission = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv', sep=',')
df_submission.head()
for i in range(df_submission.shape[0]):

    df_submission.iloc[i]['Class'] = y_classes[i]
df_submission.head()
df_submission[df_submission['ID']==398]
print(df_submission['Class'].value_counts())

df_submission.Class.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(5,5))
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

create_download_link(df_submission)