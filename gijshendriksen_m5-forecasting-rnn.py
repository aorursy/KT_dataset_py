# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import tensorflow.keras as keras



import os

import gc

import json
USING_KAGGLE = True

IS_TRAINING = True



if USING_KAGGLE:

    INPUT_DIR = '/kaggle/input'

else:

    INPUT_DIR = './input'



PLOT_DIR = './plots/'



if IS_TRAINING:

    HISTORY_DIR = './history/'

    MODEL_DIR = './models/'



    for dirname in (HISTORY_DIR, PLOT_DIR, MODEL_DIR):

        if not os.path.exists(dirname):

            os.mkdir(dirname)

else:

    HISTORY_DIR = f'{INPUT_DIR}/m5-forecasting-models/history/'

    MODEL_DIR = f'{INPUT_DIR}/m5-forecasting-models/models/'
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
PREDICTION_SIZE = 28



def compute_cum_sum(price_info, row, day_columns, week_columns):

    weekly_prices = {

        week: price_info.loc[[week, row['item_id'], row['store_id']],

            'sell_price'

        ].values[0]

        for week in np.unique(week_columns)

    }

    

    daily_sales = row[day_columns].to_numpy()

    daily_prices = [weekly_prices[week] for week in week_columns]



    return np.inner(daily_sales, daily_prices)





def compute_cum_sales(sales_info, price_info, date_info):

    day_columns = sales_info.columns[-PREDICTION_SIZE:]

    week_nos = [date_info.loc[date_info['d'] == day, 'wm_yr_wk'].values[0] for day in day_columns]



    price_info = price_info.loc[price_info['wm_yr_wk'].isin(week_nos)]

    price_info.set_index(['wm_yr_wk', 'item_id', 'store_id'], inplace=True)



    sales_info['cum_sales'] = sales_info.apply(lambda x: compute_cum_sum(price_info, x, day_columns, week_nos), axis=1)



    print(sales_info.head())
import time



start = time.time()

df_dates = reduce_mem_usage(pd.read_csv(f'{INPUT_DIR}/m5-forecasting-accuracy/calendar.csv'))

end = time.time()

print(f'Load df_dates in {end - start:.02f}s')



start = time.time()

df_prices = reduce_mem_usage(pd.read_csv(f'{INPUT_DIR}/m5-forecasting-accuracy/sell_prices.csv'))

end = time.time()

print(f'Load df_prices in {end - start:.02f}s')





COMPLETE_SALES_PATH = 'sales_complete.csv'

USE_CACHED_SALES = False



if USE_CACHED_SALES and os.path.exists(COMPLETE_SALES_PATH):

    print('Using df_sales previously computed in this session...')

    df_sales = pd.read_csv(COMPLETE_SALES_PATH)

elif USE_CACHED_SALES and os.path.exists(f'{INPUT_DIR}/m5-forecasting-models/{COMPLETE_SALES_PATH}'):

    print('Using df_sales from dataset...')

    df_sales = pd.read_csv(f'{INPUT_DIR}/m5-forecasting-models/{COMPLETE_SALES_PATH}')

else:

    start = time.time()

    df_sales = reduce_mem_usage(pd.read_csv(f'{INPUT_DIR}/m5-forecasting-accuracy/sales_train_evaluation.csv'))

    middle = time.time()

    print(f'Load df_sales in {middle - start:.02f}s')



    print('Generating cumulative sales...', end=' ')

    compute_cum_sales(df_sales, df_prices, df_dates)

    end = time.time()

    print(f'done in {end - start:.2f}s')



    df_sales.to_csv(COMPLETE_SALES_PATH)



assert 'cum_sales' in df_sales.columns
from sklearn import preprocessing



df_dates['event_type_1'] = df_dates['event_type_1'].fillna('').astype(str)

df_dates['event_type_2'] = df_dates['event_type_2'].fillna('').astype(str)



event_type_encoder = preprocessing.LabelEncoder()

event_type_encoder.fit(np.concatenate([df_dates['event_type_1'], df_dates['event_type_2']]))



event_types_1 = event_type_encoder.transform(df_dates['event_type_1'])

event_types_2 = event_type_encoder.transform(df_dates['event_type_2'])



date_features = np.swapaxes(np.array([

    event_types_1,

    event_types_2,

    df_dates['wday'],

    df_dates['month'],

    df_dates['snap_CA'],

    df_dates['snap_TX'],

    df_dates['snap_WI'],

]), 0, 1)



print(date_features.shape)



NUM_DATE_FEATURES = date_features.shape[1]
from sklearn.preprocessing import MinMaxScaler



data_scaler = MinMaxScaler(feature_range=(-1, 1))

data_np = data_scaler.fit_transform(np.swapaxes(df_sales.loc[:, 'd_1':].to_numpy(), 0, 1))



print(data_np.shape)



NUM_PRODUCTS = data_np.shape[1]
del df_prices

del df_dates



gc.collect()
class BaseGenerator(keras.utils.Sequence):

    def __init__(self, data, window_size, prediction_size, num_products):

        self.data = data

        self.window_size = window_size

        self.prediction_size = prediction_size

        self.num_products = num_products



    def __len__(self):

        return self.data.shape[0] - self.window_size - self.prediction_size + 1



    def __getitem__(self, index):

        X_sales = self.data[index:index + self.window_size].reshape((1, self.window_size, self.num_products))



        y = self.data[index + self.window_size:index + self.window_size + self.prediction_size].reshape((1, self.prediction_size, self.num_products))



        return X_sales, y





class DateGenerator(BaseGenerator):

    def __init__(self, data, date_features, window_size, prediction_size, num_products):

        super().__init__(data, window_size, prediction_size, num_products)

        self.date_features = date_features



    def __getitem__(self, index):

        X_sales, y = super().__getitem__(index)

        X_dates = np.swapaxes(self.date_features[index + self.window_size:index + self.window_size + self.prediction_size], 0, 1).reshape((1, self.prediction_size, -1))



        return (X_sales, X_dates), y



class Dataset:

    def __init__(self, data, window_size, prediction_size, num_products, date_features=None, train_test_ratio=0.75):

        x_test = np.expand_dims(data[-window_size - prediction_size:-prediction_size, :], 0)

        y_test = data[-prediction_size:, :]



        x_evaluation = np.expand_dims(data[-window_size:, :], 0)

        

        num_training = int((data.shape[0] - window_size - prediction_size) * train_test_ratio)

        

        data_train = data[:num_training, :]

        data_validation = data[num_training:-window_size - prediction_size, :]



        if date_features is None:

            self.train_generator = BaseGenerator(data_train, window_size, prediction_size, num_products)

            self.validation_generator = BaseGenerator(data_validation, window_size, prediction_size, num_products)



            self.x_test = x_test

            self.y_test = y_test

            self.x_evaluation = x_evaluation

        else:

            date_features_train = date_features[:num_training, :]

            date_features_validation = date_features[num_training:num_training + data_validation.shape[0], :]



            x_dates_test = np.expand_dims(date_features[-2 * prediction_size:-prediction_size, :], 0)

            x_dates_evaluation = np.expand_dims(date_features[-prediction_size:, :], 0)



            self.train_generator = DateGenerator(data_train, date_features_train, window_size, prediction_size, num_products)

            self.validation_generator = DateGenerator(data_validation, date_features_validation, window_size, prediction_size, num_products)



            self.x_test = (x_test, x_dates_test)

            self.y_test = y_test

            self.x_evaluation = (x_evaluation, x_dates_evaluation)
import scipy.sparse

from tensorflow.keras import losses

import time





class WRMSELoss(losses.Loss):

    def __init__(self, sales_info, name='WRMSE', use_cached=True):

        super().__init__(name=name)

        self.sales_info = sales_info

        self.level_keys = [

            (['item_id', 'store_id'], 30490),

            (['item_id', 'state_id'], 9147),

            (['item_id'], 3049),

            (['store_id', 'dept_id'], 70),

            (['store_id', 'cat_id'], 30),

            (['state_id', 'dept_id'], 21),

            (['state_id', 'cat_id'], 9),

            (['dept_id'], 7),

            (['cat_id'], 3),

            (['store_id'], 10),

            (['state_id'], 3),

            (None, 1),

        ]

        self.use_cached = use_cached



        self.contribution_matrix = self.time_computation(self.compute_contribution_matrix, 'Computing contribution matrix')

        self.weights = self.time_computation(self.compute_weights, 'Computing weights')



        self.contribution_matrix_tf = self.time_computation(lambda: self.csr_matrix_to_tf(self.contribution_matrix), 'Converting contribution matrix')

        self.weights_tf = self.time_computation(lambda: tf.convert_to_tensor(self.weights, dtype=tf.float32), 'Converting weights')



    @staticmethod

    def time_computation(computation, label):

        print(f'{label}...', end=' ')

        start = time.time()

        result = computation()

        end = time.time()

        print(f'done in {end - start:.2f}s')



        return result



    def compute_contribution_matrix(self):

        file_path = 'contribution_matrix.npz'



        if self.use_cached and os.path.exists(file_path):

            csr = scipy.sparse.load_npz(file_path)

        elif self.use_cached and os.path.exists(f'{INPUT_DIR}/m5-forecasting-models/{file_path}'):

            csr = scipy.sparse.load_npz(f'{INPUT_DIR}/m5-forecasting-models/{file_path}')

        else:

            num_lowest_level = self.sales_info.shape[0]



            contribution_matrix = [

                np.identity(num_lowest_level),

            ]



            for keys, amount in self.level_keys[1:-1]:

                groups = self.sales_info.groupby(keys)

                assert len(groups) == amount, f'Found {len(groups)} groups instead of {amount}'



                new_layer = np.zeros((amount, num_lowest_level))



                for index, (group_keys, group_df) in enumerate(groups):

                    ids = group_df.index.tolist()

                    new_layer[index, ids] = 1



                contribution_matrix.append(new_layer)



            contribution_matrix.append(np.ones((1, num_lowest_level)))



            concatenated = np.concatenate(contribution_matrix, axis=0)

            csr = scipy.sparse.csr_matrix(concatenated)

            

            del concatenated

            gc.collect()

            

            scipy.sparse.save_npz(file_path, csr)



        return scipy.sparse.csr_matrix(csr)



    @staticmethod

    def csr_matrix_to_tf(csr_matrix):

        coo = csr_matrix.tocoo()

        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.cast(tf.SparseTensor(indices, coo.data, coo.shape), tf.float32)



    def compute_weights(self):

        prices = self.sales_info['cum_sales'].to_numpy()

        leveled_prices = self.contribution_matrix * prices



        total_sum = sum(prices[:self.level_keys[0][1]])



        weights = []



        start = 0

        for key, amount in self.level_keys:

            end = start + amount



            current_prices = leveled_prices[start:end]

            weights.append(current_prices / (len(self.level_keys)) / total_sum)



            start = end



        return np.concatenate(weights, axis=0)



    def call(self, y_true, y_pred):

        y_true_full = tf.sparse.sparse_dense_matmul(self.contribution_matrix_tf, tf.transpose(tf.squeeze(y_true)))

        y_pred_full = tf.sparse.sparse_dense_matmul(self.contribution_matrix_tf, tf.transpose(tf.squeeze(y_pred)))



        rmse = tf.sqrt(tf.reduce_sum(tf.math.squared_difference(y_true_full, y_pred_full), axis=1)) / PREDICTION_SIZE



        loss = tf.tensordot(rmse, self.weights_tf, 1)



        return loss
from tensorflow.keras import models, layers





def create_simple_model(window_size, predictions_size, num_products, state_size=512, regularizer=None):

    model_input = layers.Input(shape=(window_size, num_products))

    encoder_decoder = layers.LSTM(state_size, kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(model_input)

    encoder_decoder = layers.RepeatVector(predictions_size)(encoder_decoder)

    encoder_decoder = layers.LSTM(state_size, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(encoder_decoder)

    encoder_decoder = layers.TimeDistributed(layers.Dense(num_products, kernel_regularizer=regularizer, bias_regularizer=regularizer))(encoder_decoder)

    

    model = models.Model(inputs=model_input, outputs=encoder_decoder)



    return model





def create_model_with_dates(window_size, predictions_size, num_products, num_date_features, state_size=512, regularizer=None):

    input_encoder_decoder = layers.Input(shape=(window_size, num_products))

    encoder_decoder = layers.LSTM(state_size, kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(input_encoder_decoder)

    encoder_decoder = layers.RepeatVector(predictions_size)(encoder_decoder)

    encoder_decoder = layers.LSTM(state_size, return_sequences=True, kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(encoder_decoder)

    encoder_decoder = models.Model(inputs=input_encoder_decoder, outputs=encoder_decoder)

    

    input_date_branch = layers.Input(shape=(predictions_size, num_date_features))



    output = layers.concatenate([encoder_decoder.output, input_date_branch])

    output = layers.TimeDistributed(layers.Dense(num_products, kernel_regularizer=regularizer, bias_regularizer=regularizer))(output)



    model = models.Model(inputs=[encoder_decoder.input, input_date_branch], outputs=output)



    return model
from tensorflow.keras import regularizers



WINDOW_SIZE = 100

WINDOW_SIZE_LARGE = 200

PREDICTION_SIZE = 28



MSE_LABEL = 'mean_squared_error'

MSE_LOSS = MSE_LABEL

WRMSE_LABEL = 'wrmse'

WRMSE_LOSS = WRMSELoss(df_sales, WRMSE_LABEL)



REGULARIZER = regularizers.l2()



base_dataset = Dataset(data_np, WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS)

dates_dataset = Dataset(data_np, WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, date_features=date_features)

large_dataset = Dataset(data_np, WINDOW_SIZE_LARGE, PREDICTION_SIZE, NUM_PRODUCTS)



ALL_MODELS = {

    'base_model': {

        'model': create_simple_model(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, state_size=100),

        'dataset': base_dataset,

        'loss_name': MSE_LABEL,

        'loss': MSE_LOSS,

        'name': 'Basic model (MSE loss)',

    },

    'large_model_mse': {

        'model': create_simple_model(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS),

        'dataset': base_dataset,

        'loss_name': MSE_LABEL,

        'loss': MSE_LOSS,

        'name': 'Large model (MSE loss)',

    },

    'large_model': {

        'model': create_simple_model(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS),

        'dataset': base_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss)',

    },

    'large_model_regularizer': {

        'model': create_simple_model(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, regularizer=REGULARIZER),

        'dataset': base_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + regularizer)',

    },

    'large_model_large_window': {

        'model': create_simple_model(WINDOW_SIZE_LARGE, PREDICTION_SIZE, NUM_PRODUCTS),

        'dataset': large_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + large window)',

    },

    'large_model_large_window_regularizer': {

        'model': create_simple_model(WINDOW_SIZE_LARGE, PREDICTION_SIZE, NUM_PRODUCTS, regularizer=REGULARIZER),

        'dataset': large_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + large window + regularizer)',

    },

    'large_model_dates': {

        'model': create_model_with_dates(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, NUM_DATE_FEATURES),

        'dataset': dates_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + date features)',

    },

    'large_model_dates_regularizer': {

        'model': create_model_with_dates(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, NUM_DATE_FEATURES, regularizer=REGULARIZER),

        'dataset': dates_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + date features + regularizer)',

    },

    'large_model_large_window_dates_regularizer': {

        'model': create_model_with_dates(WINDOW_SIZE, PREDICTION_SIZE, NUM_PRODUCTS, NUM_DATE_FEATURES, regularizer=REGULARIZER),

        'dataset': dates_dataset,

        'loss_name': WRMSE_LABEL,

        'loss': WRMSE_LOSS,

        'name': 'Large model (custom WRMSE loss + large window + date features + regularizer)',

    },

}
from tensorflow.keras import optimizers, callbacks





def train_model(slug, model):

    print(f'Training model: "{model["name"]}"')



    model['model'].summary()

    model['model'].compile(loss=model['loss'], optimizer=optimizers.Adam(0.001), metrics=[MSE_LOSS, WRMSE_LOSS])

    

    val_metric_name = f'val_{model["loss_name"]}'

    model_location = f'{MODEL_DIR}{slug}.h5'



    early_stopping = callbacks.EarlyStopping(monitor=val_metric_name, patience=15, mode='min', verbose=1)

    model_checkpoint = callbacks.ModelCheckpoint(model_location, monitor=val_metric_name, save_best_only=True, save_weights_only=True, mode='min', verbose=1)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor=val_metric_name, patience=4, mode='min', verbose=1)



    history = model['model'].fit_generator(

        model['dataset'].train_generator,

        validation_data=model['dataset'].validation_generator,

        epochs=50,

        callbacks=[early_stopping, model_checkpoint, reduce_lr]

    )



    hist_df = pd.DataFrame(history.history)

    hist_df.to_csv(f'{HISTORY_DIR}{slug}.csv', index=False)
tf.keras.backend.set_floatx('float32')



if IS_TRAINING:

    for slug, model in ALL_MODELS.items():

        train_model(slug, model)
import matplotlib.pyplot as plt



def plot_learning_curve(slug, model):

    df_history = pd.read_csv(f'{HISTORY_DIR}{slug}.csv')



    epochs = list(range(1, len(df_history.index) + 1))



    plt.plot(epochs, df_history['loss'], label='Training loss')

    plt.plot(epochs, df_history['val_loss'], label='Validation loss')

    plt.plot(epochs, df_history['wrmse'], label='Training WRMSE')

    plt.plot(epochs, df_history['val_wrmse'], label='Validation WRMSE')

    plt.legend()

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title(model["name"])



    plt.show()

    plt.savefig(f'{PLOT_DIR}{slug}.png')
for slug, model in ALL_MODELS.items():

    plot_learning_curve(slug, model)
for slug, model in ALL_MODELS.items():

    model['model'].load_weights(f'{MODEL_DIR}{slug}.h5')



    actual = tf.convert_to_tensor(data_scaler.inverse_transform(model['dataset'].y_test), dtype=tf.float32)

    prediction = tf.convert_to_tensor(data_scaler.inverse_transform(model['model'].predict(model['dataset'].x_test).squeeze()), dtype=tf.float32)



    score = WRMSE_LOSS.call(actual, prediction).numpy()



    print(model['name'], 'WRMSE:', score)
BEST_MODEL = 'large_model_large_window_regularizer'



model = ALL_MODELS[BEST_MODEL]

model['model'].load_weights(f'{MODEL_DIR}{BEST_MODEL}.h5')



model['model'].summary()



y_validation = data_scaler.inverse_transform(model['model'].predict(model['dataset'].x_test).squeeze()).T

y_evaluation = data_scaler.inverse_transform(model['model'].predict(model['dataset'].x_evaluation).squeeze()).T



y_full = np.concatenate([y_validation, y_evaluation], axis=0)



pred_df = pd.DataFrame(y_full, columns=[f'F{i}' for i in range(1, PREDICTION_SIZE + 1)])



evaluation_ids = df_sales['id'].values

validation_ids = [i.replace('evaluation', 'validation') for i in evaluation_ids]

all_ids = np.concatenate([validation_ids, evaluation_ids])



pred_df.insert(0, 'id', all_ids)



print(pred_df.head())



pred_df.to_csv('submission.csv', index=False)