import numpy as np 

import pandas as pd 

import keras
from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.datasets import boston_housing

from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error

from keras.regularizers import l2
df_train =pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",encoding="utf-8",index_col=0)

df_train.head()
df_test =pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",encoding="utf-8",index_col=0)

df_test.head()
df_train.isnull().any()
df_test.isnull().any()
df_train = df_train.replace("?",np.NaN)

df_test = df_test.replace("?",np.NaN)
df_train.dtypes
df_train = df_train[~df_train['Number of Quantities'].isnull()]

df_train[['Number of Quantities']] = df_train[['Number of Quantities']].astype(int)
df_test = df_test[~df_test['Number of Quantities'].isnull()]

df_test[['Number of Quantities']] = df_test[['Number of Quantities']].astype(int)
df_train = df_train[~df_train['Number of Insignificant Quantities'].isnull()]

df_train[['Number of Insignificant Quantities']] = df_train[['Number of Insignificant Quantities']].astype(int)



df_train = df_train[~df_train['Total Number of Words'].isnull()]

df_train[['Total Number of Words']] = df_train[['Total Number of Words']].astype(int)



df_train = df_train[~df_train['Number of Special Characters'].isnull()]

df_train[['Number of Special Characters']] = df_train[['Number of Special Characters']].astype(int)



df_train = df_train[~df_train['Difficulty'].isnull()]

df_train[['Difficulty']] = df_train[['Difficulty']].astype(float)

df_test = df_test[~df_test['Number of Insignificant Quantities'].isnull()]

df_test[['Number of Insignificant Quantities']] = df_test[['Number of Insignificant Quantities']].astype(int)



df_test = df_test[~df_test['Total Number of Words'].isnull()]

df_test[['Total Number of Words']] = df_test[['Total Number of Words']].astype(int)



df_test = df_test[~df_test['Number of Special Characters'].isnull()]

df_test[['Number of Special Characters']] = df_test[['Number of Special Characters']].astype(int)



df_test = df_test[~df_test['Difficulty'].isnull()]

df_test[['Difficulty']] = df_test[['Difficulty']].astype(float)
df_train['Size'].value_counts()
df_train = pd.get_dummies(data=df_train,columns=['Size'])
df_test = pd.get_dummies(data=df_test,columns=['Size'])
df_train.dtypes
df_test.dtypes
df_train.isnull().any()
df_test.isnull().any()
X = df_train.drop("Class", axis= 1)
X_test = df_test
y = df_train['Class']
df_train.columns
numerical_features = ['Number of Quantities', 'Number of Insignificant Quantities',

       'Total Number of Words', 'Total Number of Characters',

       'Number of Special Characters', 'Number of Sentences', 'First Index',

       'Second Index', 'Difficulty', 'Score']
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])
input_dim=X.shape[1]

print(input_dim)
X.shape, y.shape
cat = keras.utils.to_categorical(y)
X.shape, cat.shape
model = Sequential()
'''model = Sequential()

model.add(Dense(1,input_dim=13, activation='sigmoid'))

model.add(Dense(6,activation='sigmoid'))'''
model = Sequential()

model.add(Dense(64,input_dim=13, activation='tanh', kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.01))

model.add(Dense(32, activation='tanh',kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.01)

model.add(Dense(16, activation='tanh',kernel_regularizer=l2(0.001)))

#model.add(Dropout(rate=0.01)

model.add(Dense(8, activation='tanh',kernel_regularizer=l2(0.001)))

model.add(Dropout(rate=0.01))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 50)

es
history=model.fit(X, cat , validation_split=0.2, epochs=100, batch_size=20)
model.evaluate(X,cat)
y_pred = model.predict(X_test)

y_pred
y_classes = y_pred.argmax(axis=-1)
y_classes
df_test =pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv")
sub=pd.DataFrame()

sub['ID']=df_test['ID']

sub['Class']=y_classes
sub.head(8)
df = sub
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