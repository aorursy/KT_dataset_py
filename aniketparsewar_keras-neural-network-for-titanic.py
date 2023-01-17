import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



dataset_train = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv')



def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



dataset_train = transform_features(dataset_train)

dataset_test = transform_features(dataset_test)



dataset_train.head()





from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

dataset_train, dataset_test = encode_features(dataset_train, dataset_test)

dataset_train.head()







#Splitting data

from sklearn.model_selection import train_test_split



X_dataset_train = dataset_train.drop(['Survived', 'PassengerId'], axis=1)

y_dataset_train = dataset_train['Survived']



X = X_dataset_train.iloc[:].values

y = np.array(y_dataset_train)





num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=20)







# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(rate = 0.1))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)







y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)





# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


