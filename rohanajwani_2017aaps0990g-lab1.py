import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import sklearn.preprocessing as sk

import seaborn as sns

from keras.models import Model, Sequential

from keras.layers import Dense, Dropout

from keras.utils.np_utils import to_categorical

from keras import regularizers
train_data = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')

train_data=train_data.drop_duplicates()

x_train = train_data.drop(['Class','ID'],axis=1)

y_train = train_data.Class

for i in range(371):

    if(x_train.Size[i]=='Small'):

        x_train.Size[i]=int(-1)

    elif(x_train.Size[i]=='Medium'):

        x_train.Size[i]=int(0)

    elif(x_train.Size[i]=='Big'):

        x_train.Size[i]=int(1)
for i in range(x_train.shape[0]):

    if x_train['Number of Quantities'][i]=='?':

        x_train['Number of Quantities'][i]=int(0)

    elif x_train['Number of Insignificant Quantities'][i]=='?':

        x_train['Number of Insignificant Quantities'][i]=int(0)

    elif x_train['Size'][i]=='?':

        x_train['Size'][i]=int(0)

    elif x_train['Total Number of Words'][i]=='?':

        x_train['Total Number of Words'][i]=int(0)

    elif x_train['Total Number of Characters'][i]=='?':

        x_train['Total Number of Characters'][i]=int(0)

    elif x_train['Number of Special Characters'][i]=='?':

        x_train['Number of Special Characters'][i]=int(0)

    elif x_train['Number of Sentences'][i]=='?':

        x_train['Number of Sentences'][i]=int(0)

    elif x_train['First Index'][i]=='?':

        x_train['First Index'][i]=int(0)

    elif x_train['Second Index'][i]=='?':

        x_train['Second Index'][i]=int(0)

    elif x_train['Difficulty'][i]=='?':

        x_train['Difficulty'][i]=int(0)
y_train_onehot = to_categorical(y_train)
test_data = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')
x_test = test_data.drop('ID',axis=1)



for i in range(x_test.shape[0]):

    if(x_test.Size[i]=='Small'):

        x_test.Size[i]=int(-1)

    elif(x_test.Size[i]=='Medium'):

        x_test.Size[i]=int(0)

    elif(x_test.Size[i]=='Big'):

        x_test.Size[i]=int(1)
for i in range(x_test.shape[0]):

    if x_test['Number of Quantities'][i]=='?':

        x_test['Number of Quantities'][i]=int(0)

    elif x_test['Number of Insignificant Quantities'][i]=='?':

        x_test['Number of Insignificant Quantities'][i]=int(0)

    elif x_test['Size'][i]=='?':

        x_test['Size'][i]=int(0)

    elif x_test['Total Number of Words'][i]=='?':

        x_test['Total Number of Words'][i]=int(0)

    elif x_test['Total Number of Characters'][i]=='?':

        x_test['Total Number of Characters'][i]=int(0)

    elif x_test['Number of Special Characters'][i]=='?':

        x_test['Number of Special Characters'][i]=int(0)

    elif x_test['Number of Sentences'][i]=='?':

        x_test['Number of Sentences'][i]=int(0)

    elif x_test['First Index'][i]=='?':

        x_test['First Index'][i]=int(0)

    elif x_test['Second Index'][i]=='?':

        x_test['Second Index'][i]=int(0)

    elif x_test['Difficulty'][i]=='?':

        x_test['Difficulty'][i]=int(0)
scaler = sk.RobustScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)

x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
model = Sequential()

model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))

model.add(Dense(50, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train.values, y_train_onehot, batch_size=5, validation_split=0.2, epochs=30)
def decode(data):

    return np.argmax(data)
y_test_onehot=model.predict(x_test.values)

y_test = np.zeros(x_test.shape[0])

for i in range(x_test.shape[0]):

    y_test[i] = decode(y_test_onehot[i])

for i in range(x_test.shape[0]):

    y_test[i] = int(y_test[i])

s_y = y_test.shape[0]

x = test_data.ID

x = x.values

s_x = x.shape[0]

x = np.reshape(x,(s_x,1))

y_test = np.reshape(y_test,(s_y,1))
data = np.zeros((x.shape[0],2), dtype='int')

for i in range(x.shape[0]):

    data[i][0] = int(x[i])

    data[i][1] = int(y_test[i])



df = pd.DataFrame(data, columns = ['ID', 'Class'])

print(df)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(train_data, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(train_data)
