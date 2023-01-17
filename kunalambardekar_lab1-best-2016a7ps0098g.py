import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from keras.utils import to_categorical

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')
train.head()
train.info()
train=train.replace({'?': np.nan})

train=train.dropna()

IDcol=train['ID']

train_data=train.drop(['ID'], axis=1)

train_data['Class']=train_data['Class'].astype('object')

train_data['Difficulty']=train_data['Difficulty'].astype('float64')

train_data['Number of Quantities']=train_data['Number of Quantities'].astype('float64')

train_data['Number of Insignificant Quantities']=train_data['Number of Insignificant Quantities'].astype('float64')

train_data['Total Number of Words']=train_data['Total Number of Words'].astype('float64')

train_data['Number of Special Characters']=train_data['Number of Special Characters'].astype('float64')

train_data.info()
non_categ_cols=[]

for x in train_data.columns:

    if train_data[x].dtype=='object':

        continue

    else:

        non_categ_cols.append(x)
data1=train_data[non_categ_cols]

data2=pd.get_dummies(train_data[['Size','Class']], prefix=['Size','Class'])

train_data_final=pd.concat([IDcol,data1,data2], axis=1)
data2=pd.get_dummies(train_data[['Size','Class']], prefix=['Size','Class'])
data2.head()
train_data_final=pd.concat([IDcol,data1,data2], axis=1)
train_data_final=train_data_final.dropna()
Y=train_data_final[['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']]

X=train_data_final.drop(['ID','Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'], axis=1)
X.head()
scaler = MinMaxScaler()

scaler.fit(X)

X1=scaler.transform(X)

X=pd.DataFrame(X1, columns=X.columns)

X.head()
len(X.columns)
Y.head()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
model = Sequential()

model.add(Dense(13, input_dim=13, activation='relu'))

model.add(Dense(13, activation='relu'))

model.add(Dense(6, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train,Y_train, batch_size=3, epochs=200, verbose=1)
model.evaluate(X_val, Y_val)
def preprocess(DATA):

    DATA=DATA.replace({'?': np.nan})

    DATA=DATA.dropna()

    IDCOL=DATA['ID']

    train_data = DATA.drop(['ID'], axis=1)

    train_data['Difficulty']=train_data['Difficulty'].astype('float64')

    train_data['Number of Quantities']=train_data['Number of Quantities'].astype('float64')

    train_data['Number of Insignificant Quantities']=train_data['Number of Insignificant Quantities'].astype('float64')

    train_data['Total Number of Words']=train_data['Total Number of Words'].astype('float64')

    train_data['Number of Special Characters']=train_data['Number of Special Characters'].astype('float64')

    non_categ_cols=[]

    for x in train_data.columns:

        if train_data[x].dtype=='object':

            continue

        else:

            non_categ_cols.append(x)

    data1=train_data[non_categ_cols]

    data2=pd.get_dummies(train_data[['Size']], prefix=['Size'])

    train_data_final=pd.concat([IDCOL,data1,data2], axis=1)

    return train_data_final
test_data_final=preprocess(test)
test_data_final.info()
test_data_final.head()
X_test=test_data_final.drop(['ID'], axis=1)
scaler = MinMaxScaler()

scaler.fit(X_test)

X1_test=scaler.transform(X_test)

X_test=pd.DataFrame(X1_test, columns=X_test.columns)

X_test.head()
result=model.predict(X_test)

result
class_labels=[i.argmax() for i in result]

class_labels
result_df=pd.DataFrame({'ID':test_data_final['ID'], 'Class':class_labels})

result_df.head()
result_df.info()
result_df.to_csv('sub7.csv', index=False)
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

create_download_link(result_df)