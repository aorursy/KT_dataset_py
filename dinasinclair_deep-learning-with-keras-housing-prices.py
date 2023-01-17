import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from numpy.random import seed

seed(17)



df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
def getPreprocessor(scaler=StandardScaler()):

    # Define feature column categories by column type

    categorical_cols = df.select_dtypes(include=['object','category']).columns.to_list()

    numeric_cols = df.select_dtypes(include='number').columns.to_list()



    # Remove the target column (SalePrice) from our feature list

    numeric_cols.remove('SalePrice')



    # Preprocessing for numerical data

    numerical_transformer = Pipeline(steps=[

        ('scaler', scaler),

        ('imputer', SimpleImputer(strategy='mean'))



    ])



    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))#, categories=onehot_categories))

    ])



    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

        transformers=[

            ('numeric', numerical_transformer, numeric_cols),

            ('categorical', categorical_transformer, categorical_cols)

        ])

    

    return preprocessor
# Grab target as y, remove target from X

train_test = df.copy()

y = train_test.SalePrice

X = train_test.drop(columns=['SalePrice'])



# Split into train, test

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state = 17)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



def baseline_model(

    layer_sizes: tuple = (200, 100), 

    optimizer: str ='adam',  

    activation: str ='relu', 

    input_dim: int = 286, 

    verbose: bool =False):

    """

    Creates an NN model with the following structure:

    

    input (length input_dim) 

    --> layer 0, with layer_sizes[0] neurons 

    --> layer 1, with layer_sizes[1] neurons 

    --> ... 

    --> output

    """

    model = Sequential()

    

    # Add input layer

    if verbose:

        print("making layer {} with {} neurons".format(0, layer_sizes[0]))

    model.add(Dense(layer_sizes[0], 

                    input_dim=input_dim, 

                    kernel_initializer='normal', 

                    activation=activation))

    

    # Add hidden layers

    for i in range(1, len(layer_sizes)):

        if verbose:

            print("making layer {} with {} neurons".format(i, layer_sizes[i]))

        model.add(Dense(layer_sizes[i], 

                        kernel_initializer='normal', 

                        activation=activation))



    # Add final layer to ensure output is 1 dimensional

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
# Define pipeline, combining preprocessor and model

preprocessor = getPreprocessor()

model = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                                ('model', model)])



# Use pipeline to predict on validation data, plot validation vs predictions for the heck of it.

pipeline.fit(train_X,train_y)

preds = pipeline.predict(val_X)

plt.plot(preds, val_y, 'o')

plt.xlabel("predictions")

plt.ylabel("actuals")

plt.show()
history = model.fit(x=preprocessor.fit_transform(train_X), 

                    validation_data=(preprocessor.transform(val_X), val_y), 

                    y=train_y)



plt.plot(history.history['loss'][2:], label='training data loss')

plt.plot(history.history['val_loss'][2:], label='validation data los')

plt.legend()

plt.show()
val_loss = history.history['val_loss']

print("The validation loss is minimzed at epoch {}".format(np.asarray(val_loss).argmin()))
from sklearn.metrics import mean_squared_error



# taken from https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/



def repeated_evaluation(train_X, train_y, val_X, val_y, scaler, n_repeats=10):

    """

    Evaluates a model using preprocessing scaling scaler, by training the data n_repeats times.

    """

    # Definte the model

    model = KerasRegressor(build_fn=baseline_model, epochs=5, batch_size=5, verbose=0)

    pipeline = Pipeline(steps=[('preprocessor', getPreprocessor(scaler)),

                                ('model', model)])

    

    # Test the model by training it many times and recording the MSE each time.

    results = list()

    for n in range(n_repeats):

        pipeline.fit(train_X,train_y)

        # minimzation loss functions are negative so that you're still trying to maximize; 

        # have to multiply by -1 to make positive again

        mse = -1 * pipeline.score(val_X, val_y) 

        results.append(mse)

        

    return results
no_scaler = repeated_evaluation(train_X, train_y, val_X, val_y, scaler=None)

print("No scaling eval complete!")

minmax = repeated_evaluation(train_X, train_y, val_X, val_y, scaler=MinMaxScaler())

print("Minmax eval complete!")

standard = repeated_evaluation(train_X, train_y, val_X, val_y, scaler=StandardScaler())

print("Standard eval complete!")



print('Unscaled: %.3f (%.3f)' % (np.mean(no_scaler), np.std(no_scaler)))

print('MinMax: %.3f (%.3f)' % ( np.mean(standard), np.std(standard)))

print('Standardized: %.3f (%.3f)' % (np.mean(minmax), np.std(minmax)))



# plot results

results = [no_scaler, minmax, standard]

labels = ['unscaled', 'minmax', 'standardized']

plt.boxplot(results, labels=labels)

plt.ylabel('Mean Standard Error')

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

# uses some of https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/



# RandomizedSearchCV tuning parameters

optimizer = ['adam', 'rmsprop', 'adagrad']

activation = ['relu', 'sigmoid', 'tanh']

layer_sizes = [(200, 100), (400,), (100,100), (200, 200), (200, 200, 200)]

epochs = [10, 20, 30, 40, 50]

batch_size = [16, 32, 64, 128, 256]

param_dist = dict(optimizer=optimizer, 

                  activation=activation,

                 layer_sizes=layer_sizes,

                 batch_size = batch_size,

                 epochs = epochs)



# Create a grid, make a preprocessor + grid pipeline

# n_jobs = -1 means use the maximum number of cores available

grid = RandomizedSearchCV(estimator=model, 

                          param_distributions=param_dist, 

                          n_iter=100, 

                          scoring = 'neg_mean_squared_error', 

                          verbose=1, 

                          n_jobs=1)

grid_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                               ('grid', grid)])



# Fit the pipeline, which runs the grid

pipeline_result = grid_pipeline.fit(train_X,train_y)

grid_result = pipeline_result.named_steps['grid']



# Summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

rank = grid_result.cv_results_['rank_test_score']

for mean, stdev, param, rank in zip(means, stds, params, rank):

    if rank <= 10:

        print(f"RANK {rank:.0f}: {mean:.1f} ({stdev:.1f}) with: {param}")
# GridSearchCV tuning parameters

optimizer = ['adam', 'adagrad', 'rmsprop']

activation = ['relu']

layer_sizes = [(100,100), (200, 100), (200, 200), (200, 200, 200)]

epochs = [20, 30, 40, 50]

batch_size = [16, 32, 64]

param_grid = dict(optimizer=optimizer, 

                  activation=activation,

                 layer_sizes=layer_sizes,

                 batch_size = batch_size,

                 epochs = epochs)



# Create a grid, make a preprocessor + grid pipeline

grid = GridSearchCV(estimator=model, 

                    param_grid=param_grid, 

                    scoring = 'neg_mean_squared_error', 

                    verbose=1, 

                    n_jobs=1)

grid_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                               ('grid', grid)])



# Fit the pipeline, which runs the grid

pipeline_result = grid_pipeline.fit(train_X,train_y)

grid_result = pipeline_result.named_steps['grid']



# Summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

rank = grid_result.cv_results_['rank_test_score']

for mean, stdev, param, rank in zip(means, stds, params, rank):

    if rank <= 10:

        print(f"RANK {rank:.0f}: {mean:.1f} ({stdev:.1f}) with: {param}")
# Create predictions to be submitted!

final_predictions = pipeline_result.predict(test)

pd.DataFrame({'Id': test.Id, 'SalePrice': final_predictions}).to_csv('Keras.csv', index =False)  

print("Done :D")