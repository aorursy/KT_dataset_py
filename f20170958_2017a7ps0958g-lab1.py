import numpy as np

import pandas as pd



from keras.layers import Dense, Dropout

from keras.models import Sequential



from keras.wrappers.scikit_learn import KerasClassifier
data=pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

data.head()
def preprocess(df):

    df.drop_duplicates(inplace = True)

    df.replace("?", np.nan, inplace = True)

    df = df.dropna()

    df = df.drop(['ID'], axis = 1)

    df['Size'] = df['Size'].map({

        'Small': 0, 

        'Medium': 1,

        'Big': 2

        })

    #df = pd.get_dummies(df, columns=["Size"], prefix=["Size"] )

    return df
train = preprocess(data)



X = train.drop('Class',axis = 1)

y = train['Class']
model = Sequential()

model.add(Dense((2*X.shape[1]),input_dim=X.shape[1], activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense((2*X.shape[1]), activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



X_train = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

y_train = pd.get_dummies(y)
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, validation_split=0.3, epochs=100,batch_size = 20,verbose=1)
test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')

test1 = preprocess(test) 



scaler=StandardScaler()

X_test = pd.DataFrame(scaler.fit_transform(test1), columns=test1.columns)
y_res1 = model.predict(X_test)



y_res = y_res1.argmax(axis=1)

y_res
submission = pd.DataFrame({'ID':test['ID'],'Class':y_res})

submission.describe()
submission.to_csv('Sub_11.csv',index=False)
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(submission)