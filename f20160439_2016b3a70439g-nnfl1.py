import numpy as np

import pandas as pd

import keras

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.regularizers import l2



from sklearn.metrics import mean_absolute_error
data=pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",index_col=0)

#data = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

data.head()

df_test=pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",encoding="utf-8",index_col=0)

df_test.head()
len(data.columns)
data=data.replace('?',np.NaN)

df_test=df_test.replace('?',np.NaN)
data.dtypes
#convert object to int and float

#data['Number of Quantities']=data['Number of Quantities'].astype('int')

data = data[~data['Difficulty'].isnull()]

data['Difficulty']=data['Difficulty'].astype(float)







data = data[~data['Number of Quantities'].isnull()]

data['Number of Quantities'] = data['Number of Quantities'].astype(int)



data = data[~data['Number of Insignificant Quantities'].isnull()]

data['Number of Insignificant Quantities'] = data['Number of Insignificant Quantities'].astype(int)



data = data[~data['Total Number of Words'].isnull()]

data['Total Number of Words'] = data['Total Number of Words'].astype(int)

                                    

data = data[~data['Number of Special Characters'].isnull()]

data['Number of Special Characters'] = data['Number of Special Characters'].astype(int)



df_test = df_test[~df_test['Difficulty'].isnull()]

df_test['Difficulty']=df_test['Difficulty'].astype(float)







df_test = df_test[~df_test['Number of Quantities'].isnull()]

df_test['Number of Quantities'] = df_test['Number of Quantities'].astype(int)



df_test = df_test[~df_test['Number of Insignificant Quantities'].isnull()]

df_test['Number of Insignificant Quantities'] = df_test['Number of Insignificant Quantities'].astype(int)



df_test = df_test[~df_test['Total Number of Words'].isnull()]

df_test['Total Number of Words'] = df_test['Total Number of Words'].astype(int)

                                    

df_test = df_test[~df_test['Number of Special Characters'].isnull()]

df_test['Number of Special Characters'] = df_test['Number of Special Characters'].astype(int)
data.columns[data.isnull().any()]
corr = data.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
null_columns = df_test.columns[df_test.isnull().any()]

null_columns
data.head()
data.dtypes
df = data.copy()

df= pd.get_dummies(data=df, columns=['Size'])



df_test=pd.get_dummies(data=df_test,columns=['Size'])

df.head()
df.isnull().any()
df_test.isnull().any()
X=df.drop("Class",axis=1)

#X=X.drop("Number of Insignificant Quantities",axis=1)

#X=df.drop("Number of Insignificant Quantities",axis=1)

X_test=df_test  #.drop("Number of Insignificant Quantities",axis=1)

y=df['Class']

X.shape,X_test.shape,y.shape
df.columns
num_f=['Number of Quantities', 

       'Total Number of Words', 'Total Number of Characters',

       'Number of Special Characters', 'Number of Sentences', 'First Index',

       'Second Index', 'Difficulty', 'Score']
from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

X[num_f]=scaler.fit_transform(X[num_f])

X_test[num_f]=scaler.fit_transform(X_test[num_f])
cat=keras.utils.to_categorical(y)
X.shape,cat.shape
model = Sequential()

#model.add(Dense(128,input_dim=13, activation='relu',kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.2))

#model.add(Dense(64, activation='relu'))

#model.add(Dropout(rate=0.3))

model.add(Dense(64,input_dim=13, activation='tanh',kernel_regularizer=l2(0.001)))

model.add(Dense(32, activation='tanh',kernel_regularizer=l2(0.001)))

model.add(Dense(16, activation='tanh',kernel_regularizer=l2(0.001)))

model.add(Dense(8,activation='tanh',kernel_regularizer=l2(0.001)))

model.add(Dropout(rate=0.01))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)

es
history=model.fit(X, cat, validation_split=0.2, epochs=100,batch_size=20,callbacks=[es])
model.evaluate(X,cat)
y_pred=model.predict(X_test)
y_pred
#y_classes
model1 = Sequential()

#model.add(Dense(128,input_dim=13, activation='relu',kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.2))

#model.add(Dense(64, activation='relu'))

#model.add(Dropout(rate=0.3))

model1.add(Dense(64,input_dim=13, activation='relu',kernel_regularizer=l2(0.001)))

model1.add(Dense(32, activation='relu',kernel_regularizer=l2(0.001)))

model1.add(Dense(16, activation='relu',kernel_regularizer=l2(0.001)))

model1.add(Dense(8,activation='relu'))

model1.add(Dropout(rate=0.01))

model1.add(Dense(6,activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)

es
history=model1.fit(X, cat, validation_split=0.2, epochs=100,batch_size=20,callbacks=[es])
model1.evaluate(X,cat)
y_pred1=model1.predict(X_test)
model2 = Sequential()

#model.add(Dense(128,input_dim=13, activation='relu',kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.2))

#model.add(Dense(64, activation='relu'))

#model.add(Dropout(rate=0.3))

model2.add(Dense(64,input_dim=13, activation='relu'))

model2.add(Dense(32, activation='relu'))

model2.add(Dense(16, activation='relu'))

model2.add(Dense(8,activation='relu'))

model2.add(Dropout(rate=0.01))

model2.add(Dense(6,activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history=model2.fit(X, cat, validation_split=0.2, epochs=100,batch_size=20)
model2.evaluate(X,cat)

y_pred2=model2.predict(X_test)
y_classes=((y_pred+y_pred1+y_pred2)/3).argmax(axis=-1)

y_classes
df_test=pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv")
df_final=pd.DataFrame()
df_final['ID']=df_test['ID']

df_final['Class']=y_classes

df_final.head()
df=df_final
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

create_download_link(df)