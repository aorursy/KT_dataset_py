# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras import models

from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler





# test dataset

df_test = pd.read_csv('../input/test.csv')

#training dataset

df = pd.read_csv('../input/train.csv')



df.columns



df_test.shape[0] + df.shape[0]

df['Survived'].mean()

df.head(3)



df.columns[df.isna().any()]

df_test.columns[df_test.isna().any()]





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'

 

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

   
def prepare_data(df):

    

    # Fill missing values with the most common one

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Fill missing age with age median

    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fill missing fare with fare median

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    

    # Scale Age, SibSp, Parch, Fare to values in range between 0 to 1

    scaler = MinMaxScaler()

    df[['Age','SibSp','Parch','Fare']] = scaler.fit_transform(df[['Age','SibSp','Parch','Fare']])

    

    # Convert Sex string in 0 or 1

    df['Sex'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    

    

    # One hot encoding for the class

    df_class = pd.get_dummies(df['Pclass'],prefix='Class')

    df[df_class.columns] = df_class

    

    # One hot encoding for port of embarkation

    df_emb = pd.get_dummies(df['Embarked'],prefix='Emb')

    df[df_emb.columns] = df_emb

    

    # Extract titles from Name column and fill new column

    df['Title'] = df['Name'].map(lambda x: get_title(x))

     # Replace titles with Mr, Mrs or Miss

    df['Title'] = df.apply(replace_titles, axis=1)

    # One hot encoding on the converted titles

    df_title = pd.get_dummies(df['Title'],prefix='Title')

    df[df_title.columns] = df_title

    

    return 

prepare_data(df)

df.columns
# Columns of my choice that I think will provide value

columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Class_1', 'Class_2',

       'Class_3', 'Emb_C', 'Emb_Q', 'Emb_S', 'Title_Master',

       'Title_Miss', 'Title_Mr', 'Title_Mrs']
# Input created from selected columns

X = np.array(df[columns])

# Survived column

y = np.array(df['Survived'])
network = models.Sequential()

network.add(Dense(32, activation='relu'))

network.add(Dropout(rate=0.2))

network.add(Dense(16, activation='relu'))

network.add(Dropout(rate=0.2))

network.add(Dense(5, activation='relu'))

network.add(Dropout(rate=0.1))

network.add(Dense(1, activation='sigmoid'))

# Compile the model

network.compile(optimizer='adam',

                loss='binary_crossentropy',

                metrics=['accuracy'])

 

# Train and record historical metrics

history = network.fit(X, y, epochs=50, batch_size=10, verbose=0, validation_split=0.33)

 

# Plot train and test accuracy vs epoch number

plt.plot(history.history['acc'], label = 'train')

plt.plot(history.history['val_acc'], label = 'test')

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(loc='lower right')

plt.show()
prepare_data(df_test)

 

X_pred = np.array(df_test[columns])

y_pred = network.predict(X_pred)

 

y_pred = y_pred.reshape(418)

 

# Combine PassengerId with predicted Survived column

df_subm = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred})

 

# Show first 5 rows

df_subm.head()
# Convert to binary

def binary(x):

    # If prediction above 0.5 change to 1 (survived)

    if x > 0.5:

        return 1

    else:

        return 0

    

# Apply above function to all rows in Survived column 

df_subm['Survived'] = df_subm['Survived'].apply(binary)

df_subm.head(5)
time_string = time.strftime("%Y%m%d-%H%M%S")



# File name with date time

filename = 'kg_titanic_subm_' + time_string + '.csv'



# Save to csv

df_subm.to_csv(filename,index=False)



print('Saved file: ' + filename)