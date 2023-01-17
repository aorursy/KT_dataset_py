# Titanic Project WIP - Rory Gold



# Classification of Titanic dataset using a Neural Network with an attempt to optimize NN with randomized CV search.

# Preliminary feature selection and engineering performed.



# Randomized search CV code for a NN based on the work of Jason Brownlee 

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# Import Libraries for use

import numpy as np

import pandas as pd

import math 

import time



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV



from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.python.keras.constraints import maxnorm

from tensorflow.keras.optimizers import Adam



from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping
# Define functions

def numtostring(data, column):

    values = data[column].values

    value_str = []

    for i in range(0,len(values)):

        value_str.append(str(values[i]))

    data[column] = value_str

    return data



def other_nan(data, column):

    #Replace nan with 'O' for other

    values = data[column].values

    for i in range(0,len(data)):

       if pd.isnull(values[i]) == True:

           values[i] = 'O'

    data[column] = values

    return data



def get_target(data, column):

    data.dropna(axis=0, subset=[column], inplace=True)

    data_Y = data[column]

    data.drop([column], axis=1, inplace=True)

    return data, data_Y



def name_to_title(data):

    splt_fstop = '.'

    splt_comma = ', '

    title = []

    names = data.Name.values

    for name in names:

        partition_1 = name.partition(splt_fstop)[0]

        title.append(partition_1.partition(splt_comma)[2])

    data['Title'] = title

    del data['Name']

    return data



def Cabin_group(data):

    for i in range(0,len(data.Cabin)):

        if pd.isnull(data.Cabin.values[i]) == False:

            data.Cabin.values[i] = data.Cabin.values[i][0]

        if pd.isnull(data.Cabin.values[i]) == True:

            data.Cabin.values[i] = 'NA' 

    return data



def impute_num_columns(train_X, test_X):

    num_columns = [col for col in train_X.columns if train_X[col].dtype in ['float64', 'int'] ]

    imputer = SimpleImputer()

    imputed_train_X = pd.DataFrame(imputer.fit_transform(train_X[num_columns]))

    imputed_test_X = pd.DataFrame(imputer.transform(test_X[num_columns]))

    # Replace imputed column names

    imputed_train_X.columns = num_columns

    imputed_test_X.columns = num_columns

    # Reintroduce index

    imputed_train_X.index = train_X.index

    imputed_test_X.index = test_X.index

    # Replace columns in train_X with new sorted columns so can then do more feateng on them

    train_X[num_columns] = imputed_train_X[num_columns]

    test_X[num_columns] = imputed_test_X[num_columns]

    return train_X, test_X



def age_labelling(data):

    data_under16 = []

    data_under5 = []

    data_over50 = []

    for age in data.Age.values:

        if age < 16:

            data_under16.append('1')

        if age >= 16:

            data_under16.append('0')

        if age > 55:

            data_over50.append('1')

        if age <= 55:

            data_over50.append('0')

        if age < 5:

            data_under5.append('1')

        if age >= 5:

            data_under5.append('0')  

    data['AgeUnder_16'] = data_under16

    data['AgeOver_55'] = data_over50

    data['AgeUnder_5'] = data_under5

    return data



def OHE_encode(train_X, test_X, cat_columns):

    OHE_train_X = pd.get_dummies(train_X[cat_columns])   

    OHE_test_X = pd.get_dummies(test_X[cat_columns])

    # reindex OHE test values to add columns where a value is missing

    OHE_test_X = OHE_test_X.reindex(columns = OHE_train_X.columns, fill_value = 0)

    return OHE_train_X, OHE_test_X



def reset_index(data):

    data = data.reset_index()

    del data['index']

    return data



def scaling(train_X, test_X, num_columns):

    scaler = StandardScaler()

    train_num = train_X[num_columns]

    test_num = test_X[num_columns]

    train_scale = pd.DataFrame(scaler.fit_transform(train_num))

    test_scale = pd.DataFrame(scaler.transform(test_num))

    train_scale.columns = num_columns

    test_scale.columns = num_columns

    # reindex using original index

    train_scale.index = train_X.index

    test_scale.index = test_X.index

    return train_scale, test_scale



def get_important_features(train_X, test_X, train_Y, threshold):

    model = RandomForestClassifier(random_state=1, n_estimators=100)

    model.fit(train_X, train_Y)

    imp = model.feature_importances_

    df = pd.DataFrame()

    df['imp'] = imp

    df['cols'] = train_X.columns

    # Remove features with importance less than 0.01

    rem_cols = []

    for i in range(0, len(df)):

        if df.imp[i] < threshold:

            rem_cols.append(df.cols[i])

    for col in rem_cols:

        del train_X[col]

        del test_X[col]

    n_features = train_X.shape[1]

    important_features = train_X.columns

    return train_X, test_X, n_features, important_features



def OHE_encode_target(target):

    OHE_target = pd.get_dummies(target)

    OHE_target.columns = ['Perished', 'Survived']

    return OHE_target



def pandas_to_array(data):

    data_array = data.to_numpy()

    return data_array



def engineer_features(train_X, test_X, train_Y, test_Y):

    # Breakdown Name into Title

    train_X = name_to_title(train_X)

    test_X = name_to_title(test_X)

    

    # Group size feature

    train_X['GroupSize'] = train_X.SibSp + train_X.Parch

    test_X['GroupSize'] = test_X.SibSp + test_X.Parch

    

    # Cabin group

    train_X = Cabin_group(train_X)

    test_X = Cabin_group(test_X)

            

    # Select features of interest

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'GroupSize', 'Fare', 'Embarked', 'Cabin', 'Title']

    train_X = train_X[features]

    test_X = test_X[features]

    

    ''' Preprocess all data '''

    Pclass_train = train_X.Pclass

    Pclass_test = test_X.Pclass

    train_X = numtostring(train_X, 'Pclass')

    test_X = numtostring(test_X, 'Pclass')

    

    # Take care of missing numerical values using imputation, taking mean

    train_X, test_X = impute_num_columns(train_X, test_X)

    

    # Add under age or above age (binary columns)

    train_X = age_labelling(train_X)

    test_X = age_labelling(test_X)

    

    # Add Age + fare for Age_Fare

    train_X['Age_Fare'] = train_X.Age + train_X.Fare

    test_X['Age_Fare'] = test_X.Age + test_X.Fare

    

    # Multiply age by pclass

    train_X['Age_Pclass'] = train_X.Age * Pclass_train

    test_X['Age_Pclass'] = test_X.Age * Pclass_test

    

    ''' Encode Categorical columns '''

    # Replace nan values in categorical column Embarked

    train_X = other_nan(train_X, column = 'Embarked')

    test_X = other_nan(test_X, column = 'Embarked')

    return train_X, test_X, train_Y, test_Y



def encode_scale_features(train_X, test_X, train_Y, test_Y, cat_columns, num_columns):

    OHE_train_X, OHE_test_X = OHE_encode(train_X, test_X, cat_columns)

    

    # Scale numerical values

    train_scale, test_scale = scaling(train_X, test_X, num_columns)

    

    # Rejoin imputed and encoded values together

    train_X = pd.concat([train_scale, OHE_train_X], axis=1)

    test_X = pd.concat([test_scale, OHE_test_X], axis=1)

    

     # OHE train_Y so matches output format  

    train_Y = OHE_encode_target(train_Y)  

    test_Y = OHE_encode_target(test_Y)

    return train_X, test_X, train_Y, test_Y



def get_cat_num_columns_train(train_X):

    cat_columns = [col for col in train_X.columns if train_X[col].dtype == object]

    num_columns = [col for col in train_X.columns if train_X[col].dtype in ['float64', 'int', 'int64']]

    return cat_columns, num_columns



def keep_only_important_features(train_X, test_X, important_features):

    train_X = train_X[important_features]

    test_X = test_X[important_features]

    return train_X, test_X



def transfer_data_to_arrays(train_X, train_Y, test_X, test_Y):

    train_X = pandas_to_array(train_X)

    train_Y = pandas_to_array(train_Y)

    test_X = pandas_to_array(test_X)

    test_Y = pandas_to_array(test_Y)

    return train_X, train_Y, test_X, test_Y
# Import titanic dataset

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



train_original = train.copy()

test_original = test.copy()
# Remove rows with no target variable in train

train, train_Y = get_target(train, 'Survived')



# Process test values for prediction

# Make dummy test values to be processed.

test_dummy = pd.Series(np.zeros(test.shape[0],)) 

test_dummy[0] = 1



train_X_full, test_X_full, train_Y_full, test_Y_full = engineer_features(train, test, train_Y, test_dummy)

cat_columns, num_columns = get_cat_num_columns_train(train_X_full)

train_X_full, test_X_full, train_Y_full, test_Y_full = encode_scale_features(train_X_full, test_X_full, train_Y_full, test_Y_full, cat_columns, num_columns)



# Feature selection using RandomForestClassifier

train_X_full, test_X_full, n_features, important_features = get_important_features(train_X_full, test_X_full, train_Y_full, 0.01)



# Transfer data to arrays so can be used by tensorflow

train_X_full, train_Y_full, test_X_full, test_Y_full = transfer_data_to_arrays(train_X_full, train_Y_full, test_X_full, test_Y_full)
train = train_original.copy()

train, train_Y = get_target(train, 'Survived')



# Get validation set separated

train_X, val_X, train_Y, val_Y = train_test_split(train, train_Y, test_size = 0.2, random_state=1)



# Process validation dataset and include important features identified previously

train_X, val_X, train_Y, val_Y = engineer_features(train_X, val_X, train_Y, val_Y)

train_X, val_X, train_Y, val_Y = encode_scale_features(train_X, val_X, train_Y, val_Y, cat_columns, num_columns)

train_X, val_X = keep_only_important_features(train_X, val_X, important_features)



# Transfer data to arrays so can be used by tensorflow

train_X, train_Y, val_X, val_Y = transfer_data_to_arrays(train_X, train_Y, val_X, val_Y)
''' Implement Neural Network ''' 

# 2 classes, survived or did not survived

# Below features are those to be optimized

num_classes = 2

n_features = len(important_features)

batch_size = [1, 4, 8, 16, 32, 64, 128, 256]

dropout_rate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

weight_constraint = [1,2,3,4,5]

epochs = [10, 20, 30, 50]

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

optimizer = ['Adam']

neurons = [5, 10, 20, 50, 100, 150, 200, 300]

learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

seed=7

np.random.seed(seed)
# Implement Randomized search 

# Function to create 2 layer neural network

def create_model_2layer(neurons=100, dropout_rate=0.5, activation='softmax', weight_constraint=1, learning_rate=0.001):

    model = Sequential()

    model.add(Dense(neurons, input_dim= 18, activation= activation, kernel_constraint= maxnorm(weight_constraint)))

    model.add(Dropout(dropout_rate))

    model.add(Dense(neurons, activation= activation))

    model.add(Dropout(dropout_rate))

    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(lr=learning_rate)

    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model
# Create randomized search model using KerasClassifier

model = KerasClassifier(build_fn = create_model_2layer, verbose=True)



param_random_2layer = dict(neurons=neurons, dropout_rate=dropout_rate, activation=activation

                  , weight_constraint=weight_constraint, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)



# Perform randomized search

start_time = time.time()

random2l = RandomizedSearchCV(estimator = model, param_distributions=param_random_2layer, cv=5, random_state=1, n_iter=20, n_jobs=-1)

random_result2l = random2l.fit(train_X, train_Y)

print('My code took', time.time() - start_time, 'to run')

parameters_2l = random_result2l.best_params_



accuracy_2layer = random_result2l.score(val_X, val_Y)

print('Accuracy of random search', accuracy_2layer)
# Implement Early stopping with best performing model

model = create_model_2layer(neurons=parameters_2l['neurons'],

                            dropout_rate=parameters_2l['dropout_rate'],

                            activation=parameters_2l['activation'],

                            weight_constraint=parameters_2l['weight_constraint'],

                            learning_rate=parameters_2l['learning_rate'])
# Define early stopping parameters

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience=50)

callbacks_list = [checkpoint, es]



model.fit(train_X, train_Y, batch_size=parameters_2l['batch_size'], epochs=150, validation_data=(val_X, val_Y), callbacks = callbacks_list)
model_accuracy_early_stopping = model.evaluate(val_X, val_Y)

print('Accuracy of random search', model_accuracy_early_stopping)