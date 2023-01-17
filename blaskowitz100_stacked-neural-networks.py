######################################################################################
# Handle warnings
######################################################################################
import warnings
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.simplefilter(action='ignore', category=FutureWarning) # Scipy warnings

######################################################################################
# Standard imports
######################################################################################
import shutil
import os
import math
from copy import copy
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from scipy.special import boxcox1p

import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers import average
from keras.layers import Input
from keras.initializers import glorot_uniform

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
# Load data from kernel eda-and-preprocessing-of-the-datasets
train_path = "../input/eda-and-preprocessing-of-the-datasets/preprocessed_train.csv"
test_path = "../input/eda-and-preprocessing-of-the-datasets/preprocessed_test.csv"

# Load the dataset with Pandas
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(train.shape)
train.head()
print(test.shape)
test.head()
# Number of  samples
n_train = train.shape[0]
n_test = test.shape[0]

# Extract the GT labels and IDs from the data
train_y = train['SalePrice']
train_ids = train['Id']
test_ids = test['Id']

train = train.drop('SalePrice', axis=1)
train = train.drop('Id', axis=1)
test = test.drop('Id', axis=1)
# Plot the results of the training
def plot_history(history):
    fig = plt.figure(figsize=(15,8))
    ax = plt.subplot(211)
    
    plt.xlabel('Epoch')
    plt.ylabel('MALE, MSLE [1000$]')
    
    # Losses
    ax.set_yscale('log')
    ax.plot(history.epoch, history.history['loss'], label='Train LOSS')
    ax.plot(history.epoch, history.history['val_loss'], label='Val LOSS')
    ax.plot(history.epoch, history.history['mean_absolute_error'], label='Train MALE')
    ax.plot(history.epoch, history.history['mean_squared_error'], label='Train MSLE')
    ax.plot(history.epoch, history.history['val_mean_squared_error'], label ='Val MSLE')
    ax.plot(history.epoch, history.history['val_mean_absolute_error'], label='Val MALE')
    plt.legend()
    
    # Plot the learning_rate
    if 'lr' in history.history:
        ax = plt.subplot(212)
        plt.ylabel('Learning rate')
        ax.plot(history.epoch, history.history['lr'], label='learning_rate')
        plt.legend()
    plt.show()
    plt.close(fig)
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
def get_callbacks(bm_path):
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=bm_path,
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=False, 
        mode='auto', 
        period=1
    )

    # Scheduling the learning rate
    lr_patience = 10
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.95,
        patience=lr_patience,
        verbose=0,
        mode='auto',
        min_delta=0.0001,
        cooldown=25,  # Min. number of epochs until next lr drop
        min_lr=0
    )
    
    return lr_scheduler, checkpointer
def get_model(input_shape, num_neurons):
    model = keras.Sequential([
        # Input layer
        keras.layers.Dense(num_neurons, activation='linear', input_shape=input_shape),
        keras.layers.Dropout(0.25),
        keras.layers.LeakyReLU(alpha=0.1),
        
        # Output layer
        keras.layers.Dense(1)
    ])
    return model
X = train.values
Y = train_y.values

BATCH_SIZE = 2
NUM_FOLDS = 5
EPOCHS = 250
HOLDOUT_DICT = {'fold{}'.format(fc) : {} for fc in range(NUM_FOLDS)}

if os.path.exists('out'):
    shutil.rmtree('out')
os.makedirs('out')

BASE_MODELS = {'m0' : 16, 'm1' : 32, 'm2' : 64, 'm3' : 128}
best_models = {mod_name : [] for mod_name in BASE_MODELS.keys()}
r2_stacked = []
mse_stacked = []
mae_stacked = []
rmsle_stacked = []

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_counter = 0
for train_idxs, test_idxs in kfold.split(X, Y):
    
    HOLDOUT_DICT['fold{}'.format(fold_counter)]['gt'] = Y[test_idxs]
    
    for model_name, model_config in BASE_MODELS.items(): 
        
        # Integrate model config
        model = get_model((train.shape[1],), num_neurons=model_config)
        
        # Be sure the weights are randomized
        reset_weights(model)

        optimizer = 'adam'
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
        
        bm_path = "out/fold{}{}_config{}.h5".format(fold_counter, model_name, model_config)

        # Get callbacks for the training process
        lr_scheduler, checkpointer = get_callbacks(bm_path)

        history = model.fit(
            x=X[train_idxs],
            y=Y[train_idxs],
            validation_data=(X[test_idxs], Y[test_idxs]),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[
                checkpointer, 
                #lr_scheduler
            ],
            verbose=0,
            shuffle=True
        )

        model.load_weights(bm_path)
        predict_val = model.predict(X[test_idxs]).flatten()
        
        HOLDOUT_DICT['fold{}'.format(fold_counter)]['{}'.format(model_name)] = predict_val.copy()
        
        r2 = r2_score(predict_val, Y[test_idxs])
        mse = mean_squared_error(predict_val, Y[test_idxs])
        mae = mean_absolute_error(predict_val, Y[test_idxs])
        rmsle = math.sqrt(mse)

        best_models[model_name].append(bm_path)
        mse_stacked.append(mse)
        mae_stacked.append(mae)
        rmsle_stacked.append(rmsle)
        r2_stacked.append(r2)

        plot_history(history)
        print("{}/{} folds model {}: MSE: {} || MAE: {} || RMSLE: {} || R2: {}".format(fold_counter + 1, NUM_FOLDS, model_name, mse, mae, rmsle, r2))

        K.clear_session()
    fold_counter += 1

print("AVG_RMSLE: {} || AVG_R2: {}".format(np.mean(rmsle_stacked), np.mean(r2_stacked)))
df_list = []
for fold, fold_data in HOLDOUT_DICT.items():
    df_list.append(pd.DataFrame(fold_data))

holdouts_df = pd.concat(df_list)
print(holdouts_df.head(3))
print(holdouts_df.shape)
from sklearn.linear_model import Lasso
meta_model = Lasso(alpha=0.0005, random_state=1)
meta_model.fit(X=holdouts_df[list(BASE_MODELS.keys())], y=holdouts_df['gt'])
rmsle = math.sqrt(mean_squared_error(meta_model.predict(holdouts_df[list(BASE_MODELS.keys())]), holdouts_df['gt']))
print(rmsle)
# Stack the best model of each fold together
def ensemble_models(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models] 
    
    # averaging outputs
    yAvg = average(yModels) 
    
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensembleModel')  
    return modelEns
ensembles = {}

for model_name, bm_paths in best_models.items():
    temp_ens_models = []
    for model_path in bm_paths:
        model_temp = load_model(model_path)
        model_temp.name = model_path
        temp_ens_models.append(model_temp)
    ens_input = keras.Input(shape=temp_ens_models[0].input_shape[1:])
    ensembles[model_name] = ensemble_models(temp_ens_models, ens_input)
    
    ensembles[model_name].save('out/ensemble_base_model_{}.h5'.format(model_name))
    print(ensembles[model_name].summary())
def metamodel_predict(ensemble_models_dict, meta_model, data):
    res = {}
    for model_name, model in ensemble_models_dict.items():
        res[model_name] = model.predict(data).flatten()
    
    res_df = pd.DataFrame(res)
    meta_res = meta_model.predict(res_df.values)
    res_df['meta'] = meta_res
    
    return res_df, meta_res
res_df, meta_res = metamodel_predict(ensembles, meta_model, X)
for col in res_df.columns:
    pred = res_df[col]
    print("Model {}: RMSLE: {} || R2: {}".format(col, math.sqrt(mean_squared_error(pred, Y)), r2_score(pred, Y)))
res_submission_df, meta_res_submission = metamodel_predict(ensembles, meta_model, test)
sub = pd.DataFrame()
sub['Id'] = test_ids
sub['SalePrice'] = np.expm1(meta_res_submission)
sub.to_csv(os.path.join('submission_sm.csv') ,index=False)

sub.head()