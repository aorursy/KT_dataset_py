from IPython.display import HTML 



import base64 # create download link

import re # relugar expression

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualizations



# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# /kaggle/temp/ won't be preserved in the output file



# Create download link for output file



def create_download_link(df, title = "Download CSV file", filename = "result_titanic.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
# Data I/O



train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



full_data = [train_data, test_data]
# Generate new features



for data in full_data:

    data['Has_Cabin'] = data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
# RE method to extract title



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



for data in full_data:

    data['Title'] = data['Name'].apply(get_title)
# Replace values -1



for data in full_data:

    # replace missing to mode

    data['Embarked'] = data['Embarked'].replace(np.nan, 'S')

    

    #replace for simplicity

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 

                                           'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data['Title'] = data['Title'].replace(np.nan, 'None')
# Replace values -2



# replace missing by knn

from sklearn.impute import KNNImputer



def nan_padding(data, nan_cols):

    for nan_col in nan_cols:

        imputer = KNNImputer(n_neighbors = 3)

        data[nan_col] = imputer.fit_transform(data[nan_col].values.reshape(-1, 1))

    return data



nan_cols = ['Age', 'SibSp', 'Parch', 'Fare']



train_data = nan_padding(train_data, nan_cols)

test_data = nan_padding(test_data, nan_cols)
# Label encoding



from sklearn.preprocessing import LabelEncoder



def as_int(data, cols_classes):

    for col_class in cols_classes:        

        le = LabelEncoder()

        le.fit(col_class[1])

        data[col_class[0]] = le.transform(data[col_class[0]]) 

    return data



cols = ['Sex', 'Embarked', 'Title']

cols_classes = []

for feature in cols:

    cols_classes.append([feature, list(data[feature].value_counts().keys())])

    

train_data = as_int(train_data, cols_classes)

test_data = as_int(test_data, cols_classes)
# Drop not concerned columns



test_passenger_Id = test_data['PassengerId']



not_concerned_cols = ['PassengerId','Ticket','Name','Cabin']



train_data = train_data.drop(not_concerned_cols, axis = 1)

test_data = test_data.drop(not_concerned_cols, axis = 1)
# Convert to dummy columns



def convert_to_dummy(data, dummy_cols):

    for dummy_col in dummy_cols:

        data = pd.concat([data, pd.get_dummies(data[dummy_col], prefix = dummy_col)], axis = 1)

        data = data.drop(dummy_col, axis = 1)

    return data



dummy_cols = ['Embarked', 'Title']



train_data = convert_to_dummy(train_data, dummy_cols)

test_data = convert_to_dummy(test_data, dummy_cols)
# Data normalization



from sklearn.preprocessing import MinMaxScaler



def normalize(data, norm_cols):

    scaler = MinMaxScaler()

    for norm_col in norm_cols:

        data[norm_col] = scaler.fit_transform(data[norm_col].values.reshape(-1,1))

    return data



norm_cols = ['Age', 'Fare', 'Pclass']



train_data = normalize(train_data, norm_cols)

test_data = normalize(test_data, norm_cols)
# Split valid test data



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



def split_valid_test(data, fraction = (1 - 0.8)):

    lb = LabelBinarizer()

    data_y = data['Survived']

    data_y = lb.fit_transform(data_y)

    data_x = data.drop(['Survived'], axis = 1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size = fraction)

    return train_x.values, valid_x, train_y, valid_y



train_x, valid_x, train_y, valid_y = split_valid_test(train_data)
# Implement neural network



import tensorflow as tf



model = tf.keras.Sequential([ 

    tf.keras.layers.Dense(64, 

                          kernel_initializer = 'glorot_normal', 

                          #bias_initializer = 'truncated_normal', 

                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 

                          bias_regularizer = tf.keras.regularizers.l2(0.01), 

                          activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, 

                          kernel_initializer = 'glorot_normal',

                          #bias_initializer = 'truncated_normal', 

                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 

                          bias_regularizer = tf.keras.regularizers.l2(0.01), 

                          activation = 'relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, 

                          kernel_initializer = 'glorot_normal',

                          #bias_initializer = 'truncated_normal', 

                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 

                          bias_regularizer = tf.keras.regularizers.l2(0.01), 

                          activation = 'relu'),

    #tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, 

                          kernel_initializer = 'glorot_normal', 

                          #bias_initializer = 'truncated_normal', 

                          kernel_regularizer = tf.keras.regularizers.l2(0.001), 

                          bias_regularizer = tf.keras.regularizers.l2(0.01), 

                          activation = 'sigmoid')

])



model.compile(optimizer='adam', 

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 

              metrics=['accuracy'])



history = model.fit(train_x, train_y, epochs=100, validation_data=(valid_x, valid_y))
# Plot model accuracy



valid_loss, valid_acc = model.evaluate(valid_x, valid_y, verbose=2)

print('\nTest accuracy:', valid_acc)



plt.figure(figsize = [8, 6])

plt.plot(history.history['accuracy'], 'r', linewidth = 3.0)

plt.plot(history.history['val_accuracy'], 'b', linewidth = 3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize = 18)

plt.xlabel('Epochs ', fontsize = 16)

plt.ylabel('Accuracy', fontsize = 16)

plt.title('Accuracy Curves', fontsize = 16)

plt.show()
# Predict



predictions = model.predict(test_data)

predictions = np.where(predictions > 0.5, 1, 0)



# Create and download output file



data_to_submit = pd.DataFrame()

data_to_submit['PassengerId'] = test_passenger_Id

data_to_submit['Survived'] = pd.DataFrame(predictions)

#data_to_submit.to_csv('csv_to_submit.csv', header=True, index=False)



create_download_link(data_to_submit)