import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')


from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

# remove all ? columns

df_filtered = df2[df2['Number of Quantities'] != '?']

df_filtered = df_filtered[df_filtered['Number of Insignificant Quantities'] != '?']

df_filtered = df_filtered[df_filtered['Size'] != '?']

df_filtered = df_filtered[df_filtered['Total Number of Words'] != '?']

df_filtered = df_filtered[df_filtered['Number of Special Characters'] != '?']

df_filtered = df_filtered[df_filtered['Difficulty'] != '?']



df_filtered.shape

df_filtered.info()
#change the type of data from objects to integers

df_filtered['Number of Quantities'] = df_filtered['Number of Quantities'].astype(int) # change datatype of column after importing



df_filtered['Number of Insignificant Quantities'] = df_filtered['Number of Insignificant Quantities'].astype(int)



df_filtered['Total Number of Words'] = df_filtered['Total Number of Words'].astype(int)



df_filtered['Number of Special Characters'] = df_filtered['Number of Special Characters'].astype(int)



df_filtered['Difficulty'] = df_filtered['Difficulty'].astype(float)



df_filtered.info()
# one hot encoding for Size

df_filtered = pd.get_dummies(df_filtered, columns=['Size'], prefix = ['Size'])

df_filtered.tail()
#drop duplicates

df_filtered=df_filtered.drop_duplicates() # returns copy of modified object
#dropping irrelevant columns

df_filtered = df_filtered.drop(['ID'],axis = 1)

df_filtered.head()
#separating class column - preparing x_train and y_train

y_train = df_filtered['Class']

x_train = df_filtered.drop(['Class'], axis = 1)

x_train.info()
# input test set



df_test = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv') 



df_test = pd.get_dummies(df_test, columns=['Size'], prefix = ['Size'])



x_test = df_test



x_test.info()
# scaler before training

from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

scaled_data=scaler.fit(x_train).transform(x_train)

scaled_df=pd.DataFrame(scaled_data,columns=x_train.columns)

scaled_df.tail()

x_train = scaled_df
# model

from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error
x_train.shape, y_train.shape
from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_train.shape
# Build the architecture

from keras.regularizers import l2



model = Sequential()

#model.add(Dense(13,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activation = 'relu')

model.add(Dense(13, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.02)))

model.add(Dense(4, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(6, activation='softmax'))



# add more layers if desired
# Compile 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train2 = x_train.values

#y_train2 = y_train.values

model.fit(x_train2, y_train, validation_split=0.4, epochs=25,batch_size=20)
x_test = x_test.drop(['ID'],axis = 1)
pred = model.predict(x_test)

labels = np.argmax(pred, axis=-1)    

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