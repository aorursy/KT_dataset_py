import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

from tensorflow.keras.models import load_model
# Input

file_train = '../input/house-prices-advanced-regression-techniques/train.csv'

file_test = '../input/house-prices-advanced-regression-techniques/test.csv'



# Output

model_output_file = 'NN_1D_CSV_model.h5'



# Model specific

target = 'SalePrice'

index_col = 'Id'



# Validation Split, to separate testing and validation data

VALIDATION_SPLIT_SIZE = 0.2
def load_raw_data(file_path):

    return pd.read_csv(file_path, index_col = index_col)
NA_THRESHOLD = 200

NA_CATEGORICAL_SUBSTITUTION = 'NULL'

NA_NUMERICAL_SUBSTITUTION = 0



def feature_extract(data):

        feature_columns = []

        # Columns with high missing values

        highna = [cname for cname in data.columns if data[cname].isna().sum() > NA_THRESHOLD]

        lowna = [cname for cname in data.columns if data[cname].isna().sum() <= NA_THRESHOLD]

        

        # Dropping columns with high number of missing values

        data = data.drop(highna, axis=1)



        # Low cardinality cols only

        categorical_cols = [cname for cname in data.columns

                                if data[cname].nunique() < 10 and data[cname].dtype == "object"]

        numeric_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]

        if target in categorical_cols: categorical_cols.remove(target)

        if target in numeric_cols: numeric_cols.remove(target)



        return feature_columns, categorical_cols, numeric_cols

    
def prep_features(data, categorical_cols, numeric_cols, x_train = None):

    empty_fill = {}

    for feature in categorical_cols:

        empty_fill[feature] = NA_CATEGORICAL_SUBSTITUTION

    for feature in numeric_cols:

        empty_fill[feature] = NA_NUMERICAL_SUBSTITUTION

    data = data.fillna(empty_fill)



    data = data[categorical_cols + numeric_cols]



    # One Hot Numeric values

    x = pd.get_dummies(data, dummy_na=True)

    if x_train is not None:

        x_train, x = x_train.align(x, join='left', axis=1)

        x = x.fillna(0)

    else:

        x_train = x



    # Normalization on numerical values

    normed_x = norm(x, x_train)



    return normed_x

    

def norm(x, x_train):

    train_stats = x_train.describe()

    train_stats = train_stats.transpose()

    normalized =  (x - train_stats['mean']) / train_stats['std']

    return normalized.fillna(0)
def generate_model (feature_len, optimizer, dropout = 0.3, units = 64, leaky = 0.05):

    model = Sequential([

        layers.Dense(units, activation='relu', input_shape=feature_len),

        layers.BatchNormalization(),

        layers.LeakyReLU(alpha=leaky),

        layers.Dropout(dropout),

        layers.Dense(1)

    ])

    model.compile(loss='mse',

                       optimizer=optimizer,

                       metrics=['mae', 'mse'])

    return model
def evaluate_model (x_train, y_train, x_val, y_val, optimizer, dropout = 0.3, units = 64, leaky = 0.05, epochs = 10):

    model = generate_model(feature_len = [len(x_train.keys())],

                           optimizer = optimizer,

                           dropout = dropout,

                           units = units,

                           leaky = leaky)

    model.fit(x = x_train,

              y = y_train,

              epochs = epochs,

              validation_split = 0.2,

              verbose = 0)

    loss, mae, mse = model.evaluate(x_val, y_val, verbose=0)

    return mae

    

def evaluate_model_parameters (x_train, y_train, x_val, y_val):

    evaluating_optimizers = [tf.keras.optimizers.RMSprop(0.001),

                             tf.keras.optimizers.Adagrad(learning_rate=0.001),

                             tf.keras.optimizers.Adadelta(0.001),

                             tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

                             ]

    evaluating_dropouts = [0.9, 0.8, 0.7, 0.6, 0.5,0.25,0.1]

    evaluating_units = [32, 64, 128]

    evaluating_leak = [0.05, 0.1, 0.01]

    evaluating_epochs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]



    # Basic Search

    best_score = float("inf")

    best_optimizer = evaluating_optimizers[0]

    for optimizer in evaluating_optimizers:

        score = evaluate_model (x_train, y_train, x_val, y_val, optimizer)

        if best_score > score:

            best_score = score

            best_optimizer = optimizer



    best_score = float("inf")

    best_dropout = evaluating_dropouts[0]

    for dropout in evaluating_dropouts:

        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,

                                dropout = dropout)

        if best_score > score:

            best_score = score

            best_dropout = dropout



    best_score = float("inf")

    best_units = evaluating_units[0]

    for units in evaluating_units:

        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,

                                dropout = best_dropout,

                                units = units)

        if best_score > score:

            best_score = score

            best_units = units



    best_score = float("inf")

    best_epochs = evaluating_epochs[0]

    for epochs in evaluating_epochs:

        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,

                                dropout = best_dropout,

                                units = best_units,

                                epochs = epochs)

        if best_score > score:

            best_score = score

            best_epochs = epochs



    best_score = float("inf")

    best_leak = evaluating_leak[0]

    for leaky in evaluating_leak:

        score = evaluate_model (x_train, y_train, x_val, y_val, best_optimizer,

                                dropout = best_dropout,

                                units = best_units,

                                leaky = leaky,

                                epochs = epochs)

        if best_score > score:

            best_score = score

            best_leak = leaky



    return best_optimizer, best_dropout, best_units, best_epochs, best_leak
# Load training data

raw_train = load_raw_data(file_train)

raw_train, raw_val = train_test_split(raw_train, test_size = VALIDATION_SPLIT_SIZE)



# Prepocessing Training Data

feature_columns, categorical_cols, numeric_cols = feature_extract(raw_train)



x_train = prep_features(raw_train, categorical_cols, numeric_cols)        

x_val = prep_features(raw_val, categorical_cols, numeric_cols, x_train)



y_train = raw_train[target]

y_val = raw_val[target]

best_optimizer, best_dropout, best_units, best_epochs, best_leak = evaluate_model_parameters(x_train,

                                                                                             y_train,

                                                                                             x_val,

                                                                                             y_val)



print ('Best optimizer: ')

print (best_optimizer)

print ('Best dropout: ' + str(best_dropout))

print ('Best units: ' + str(best_units))

print ('Best leak factor: ' + str(best_leak))

print ('Best epochs: ' + str(best_epochs))

model = generate_model(feature_len = [len(x_train.keys())],

                       optimizer = best_optimizer,

                       dropout = best_dropout,

                       units = best_units,

                       leaky = best_leak)

model.fit(x = x_train,

          y = y_train,

          epochs = best_epochs,

          validation_split = 0.2,

          verbose = 0)

loss, mae, mse = model.evaluate(x_val, y_val, verbose=2)

print ('MAE: ' + str(mae))

print ('MSE: ' + str(mse))

print ('Loss: ' + str(loss))
raw_test = load_raw_data(file_test)

x_test = prep_features(raw_test, categorical_cols, numeric_cols, x_train)

preds = model.predict(x_test).flatten()
output = pd.DataFrame({'Id': x_test.index,

                       target: preds})

output = output.fillna(y_train.mean())

output.to_csv('submission.csv', index=False)

## Saving the model

model.save(model_output_file)



## saving x_train meta data: categorical_cols, numeric_cols

x_train_json = json.dumps({"categorical": categorical_cols, "numerical": numeric_cols}, indent = 4) 

  

# Writing to sample.json 

with open("x_train.json", "w") as outfile: 

    outfile.write(json_object) 
## Loading the model

model = load_model(model_output_file)



with open('x_train.json', 'r') as openfile: 

    x_train_meta = json.load(openfile) 



preds = model.predict(x=x_test, batch_size=100, verbose=0)

print("Predictions:")

print(preds)