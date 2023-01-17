import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sell_prices.head()
sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']

sell_prices['sell_price'].describe()
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

sales.head()
sales.info()
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar.head()
calendar.info()
sales_training = sales.iloc[:,6:]

sales_training.head()
sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

sample_submission
rows = [0, 42, 1024, 10024]

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,15))

for ax, row in zip(axes, rows):

    sales_training.iloc[row].plot(ax=ax)
rows = [0, 42, 1024, 10024]

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,15))

for ax, row in zip(axes, rows):

    sales_training.iloc[row].rolling(30).mean().plot(ax=ax)
from statsmodels.tsa.stattools import adfuller
D = []



try:

    stationarity_differences = pd.read_csv('/kaggle/input/stationary-differences/stationarity_differences.csv').iloc[:,0]

except FileNotFoundError:

    for index, row in sales_training.iterrows():

        d = 0

        p_val = adfuller(row, autolag='AIC')[1]

        while p_val > 0.05:

            d += 1

            row = row.diff()[1:]

            p_val = adfuller(row, autolag='AIC')[1]

        D.append(d)

    pd.Series(D).to_csv('./stationarity_differences.csv', index=False)

    stationarity_differences = pd.read_csv('/kaggle/input/stationary-differences/stationarity_differences.csv').iloc[:,0]
stationarity_differences.value_counts()
sales_training.iloc[12791].plot()
stationary_train_sales = np.diff(sales_training.values, axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)

scaler.fit(stationary_train_sales.T)

X_train = scaler.transform(stationary_train_sales.T).T

scales = scaler.scale_
calendar
sales_normalized = calendar[['wm_yr_wk','d']].iloc[:1941]

sales_normalized = pd.DataFrame(X_train, columns=sales_normalized['d'][1:])

sales_normalized.insert(0, 'id', sales['item_id'] + '_' + sales['store_id'])

sales_normalized
rows = [0, 42, 1024, 10024]

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,15))

for ax, row in zip(axes, rows):

    sales_normalized.iloc[row, 1:].plot(ax=ax)
sales_normalized
rows = [42, 10024]

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,15))

for ax, row in zip(axes, rows):

    integrated_series = np.cumsum(sales_normalized.iloc[row, 1:]*scales[row])

    c = sales_training.iloc[row, 0]

    integrated_series = pd.Series(integrated_series + c).shift(1)

    integrated_series[:100].plot(ax=ax, style='r--', legend=True, label='re-integrated')

    sales_training.iloc[row][:100].plot(ax=ax, legend=True, label='original')

    total_numerical_error = np.abs(np.array(pd.Series(integrated_series)[1:].to_numpy() - sales_training.iloc[row,1:-1].to_numpy())).sum()

    ax.set_title('Total numerical error: {:.2f}'.format(total_numerical_error))
sns.distplot(sell_prices['id'].value_counts(), kde=False, axlabel='number of weeks the product was priced on')
sales_two_week_sum = sales_training.rolling(14, axis=1).sum()
for col in range(13):

    sales_two_week_sum.iloc[:, col] = sales_two_week_sum.iloc[:, 13]

    

is_off_the_shelf = sales_two_week_sum == 0

#to the days when the products were off for 14 last days we add those 14 days

is_off_the_shelf = is_off_the_shelf | is_off_the_shelf.shift(-13, axis=1)

is_on_the_shelf = is_off_the_shelf == False

# True/False to 1/0

# is_on_the_shelf = is_on_the_shelf.astype('int')
is_on_the_shelf
rows = [0, 42, 1024, 10024]

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,15))

for ax, row in zip(axes, rows):

    shelf = pd.DataFrame(is_on_the_shelf.iloc[row])

    shelf.columns = ['is_on_the_shelf']

    shelf['sold'] = sales_training.iloc[row]

    shelf = shelf.reset_index()

    shelf.drop('index', inplace=True, axis=1)

    shelf[shelf['is_on_the_shelf'] == True]['sold'].plot(legend=True, label='on shelf', ax=ax)

    shelf[shelf['is_on_the_shelf'] == False]['sold'].plot(style='o', legend=True, label='not on shelf', ax=ax)
sales['dept_id'].value_counts()
sales['cat_id'].value_counts()
sales['state_id'].value_counts()
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

dept_encoded = encoder.fit_transform(sales['dept_id'].values.reshape(-1,1))

cat_encoded = encoder.fit_transform(sales['cat_id'].values.reshape(-1,1))

state_encoded = encoder.fit_transform(sales['state_id'].values.reshape(-1,1))
train_prices = pd.read_csv('/kaggle/input/train-price-parser/train_prices.csv')

#sort IDs to match our order of IDs

train_prices['id'] = train_prices['id'].astype("category")

correct_order = sales['item_id'] + '_' + sales['store_id']

train_prices['id'].cat.set_categories(correct_order, inplace=True)

train_prices = train_prices.sort_values(["id"]).reset_index().drop(columns=['index'])
print('train_prices - observed {:.2f}% missing values'.format(train_prices.isnull().sum(axis=1).mean()/1970 * 100))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant',fill_value='no_event')

imputed_calendar_primary = imputer.fit_transform(calendar['event_name_1'].to_numpy().reshape(-1,1))

imputed_calendar_secondary = imputer.fit_transform(calendar['event_name_2'].to_numpy().reshape(-1,1))
imputed_calendar = np.hstack((imputed_calendar_primary,imputed_calendar_secondary))
# a quick note - this has for some reason already beed 'differenciated', by which a mean the holidays lasting for few days

# are denoted as beggining and end of the holliday

encoder = OneHotEncoder()

calendar_encoded = encoder.fit_transform(imputed_calendar)



# the line meaning that no event happens dubled so we throw one out

calendar_encoded = calendar_encoded[:,:-1]
# never forget to equally differentiate every time series!

is_on_the_shelf_diff = is_on_the_shelf.diff(axis=1).iloc[:,1:]

is_on_the_shelf_diff = is_on_the_shelf_diff.astype('int')
class M5_SeriesGenerator:

    def __init__(self):

        self.day_zero = 1941

        self.max_rows = 30490

        self.rows_remaining = np.arange(self.max_rows)

        

    def reset(self):

        self.rows_remaining = np.arange(self.max_rows)

        

    def next_batch(self, in_points=30, out_points=3, batch_size=10):

        X_batch = []

        X_past_batch = []

        scale_batch = []

        c_batch = []

        facts_batch = []

        y_batch = []

        

        for _ in range(batch_size):

            if self.rows_remaining.shape[0] == 0:

                return False, (None, None)

        

            row = np.random.randint(self.rows_remaining.shape[0])

            self.rows_remaining = np.delete(self.rows_remaining, row)

            X_train_start = self.day_zero-366-in_points

            X_prev_year_start = self.day_zero-2*365



            while is_on_the_shelf.iloc[row, X_train_start+in_points] == False:

                if self.rows_remaining.shape[0] == 0:

                    return False, (None, None)

                row = np.random.randint(self.rows_remaining.shape[0])

                self.rows_remaining = np.delete(self.rows_remaining, row)



            Xsales_train = sales_normalized.iloc[row, X_train_start+1:X_train_start+in_points+1].values.astype(np.float32)

            Xsales_prev_year = sales_normalized.iloc[row, X_prev_year_start:X_prev_year_start+out_points].values.astype(np.float32)



            Y_train_start = X_train_start+in_points

            Yprices_train = train_prices.iloc[row, Y_train_start+2:Y_train_start+out_points+2].values.astype(np.float32)

            Yevents_train = calendar_encoded[Y_train_start+1:Y_train_start+out_points+1, :].toarray().astype(int)

            Ydept_train = np.tile(dept_encoded[row].toarray().astype(int),(out_points,1))

            Ycat_train = np.tile(cat_encoded[row].toarray().astype(int),(out_points,1))

            Ystate_train = np.tile(state_encoded[row].toarray().astype(int),(out_points,1))

            Ysales_train = sales_training.iloc[row, Y_train_start+1:Y_train_start+out_points+1].values.astype(int).flatten()

            

            Yfacts = np.hstack((Yprices_train.reshape(-1, 1), Yevents_train, Ydept_train, Ycat_train, Ystate_train))

            integral_constant = sales_training.iloc[row, X_train_start+in_points]

            scale = scales[row]

            

            X_batch.append(Xsales_train.reshape(-1, 1))

            X_past_batch.append(Xsales_prev_year.reshape(-1, 1))

            scale_batch.append(scale)

            c_batch.append(integral_constant)

            facts_batch.append(Yfacts)

            y_batch.append(Ysales_train)

        return True, ((np.asarray(X_batch), np.concatenate((np.asarray(X_past_batch), np.asarray(facts_batch)), axis=2), np.asarray(scale_batch), np.asarray(c_batch)), np.asarray(y_batch))
def eval_series_data_gen(in_points = 120, out_points=28, end_of_data=1913, max_row=30490):    

    row = 0

    while row < max_row:

        X_batch = []

        X_past_batch = []

        scale_batch = []

        c_batch = []

        facts_batch = []

        y_batch = []

        X_train_start = end_of_data-1-in_points

        X_prev_year_start = end_of_data-365

        

        if is_on_the_shelf.iloc[row, X_train_start+in_points] == False:

            row += 1

            yield False, None

        else:

            Xsales_train = sales_normalized.iloc[row, X_train_start+1:X_train_start+in_points+1].values.astype(np.float32)

            Xsales_prev_year = sales_normalized.iloc[row, X_prev_year_start:X_prev_year_start+out_points].values.astype(np.float32)



            Y_train_start = X_train_start+in_points

            Yprices_train = train_prices.iloc[row, Y_train_start+2:Y_train_start+out_points+2].values.astype(np.float32)

            Yevents_train = calendar_encoded[Y_train_start+1:Y_train_start+out_points+1, :].toarray().astype(int)

            Ydept_train = np.tile(dept_encoded[row].toarray().astype(int),(out_points,1))

            Ycat_train = np.tile(cat_encoded[row].toarray().astype(int),(out_points,1))

            Ystate_train = np.tile(state_encoded[row].toarray().astype(int),(out_points,1))

            Ysales_train = sales_training.iloc[row, Y_train_start+1:Y_train_start+out_points+1].values.astype(int).flatten()



            Yfacts = np.hstack((Yprices_train.reshape(-1, 1), Yevents_train, Ydept_train, Ycat_train, Ystate_train))

            integral_constant = sales_training.iloc[row, X_train_start+in_points]

            scale = scales[row]



            X_batch.append(Xsales_train.reshape(-1, 1))

            X_past_batch.append(Xsales_prev_year.reshape(-1, 1))

            scale_batch.append(scale)

            c_batch.append(integral_constant)

            facts_batch.append(Yfacts)

            y_batch.append(Ysales_train)

            row += 1

            yield True, (np.asarray(X_batch), np.concatenate((np.asarray(X_past_batch), np.asarray(facts_batch)), axis=2), np.asarray(scale_batch), np.asarray(c_batch))
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import BatchNormalization
class M5_Net(keras.Model):

    def __init__(self, input_timesteps, output_timesteps, batch_size=1):

        super(M5_Net, self).__init__()

        self.input_timesteps = input_timesteps

        self.output_timesteps = output_timesteps

        self.batch_size = batch_size



        self.gru1 = tf.keras.layers.GRU(32, return_sequences=True)

        self.gru1a = tf.keras.layers.GRU(64, return_sequences=True)

        self.gru2 = tf.keras.layers.GRU(64, return_sequences=True)

        self.gru2a = tf.keras.layers.GRU(32, return_sequences=True)

        self.gru_out = tf.keras.layers.GRU(1, return_sequences=True)

        self.dense1 = keras.layers.Dense(self.output_timesteps, activation="selu", kernel_initializer="lecun_normal")

        

    def call(self, input_data):

        series_data, historical_data, scale, integral_constant = input_data

        

        x = BatchNormalization()(self.gru1(series_data))

        x = BatchNormalization()(self.gru1a(x))

        x = tf.reshape(x, [self.batch_size, -1])

        x = BatchNormalization()(self.dense1(x))

        x = tf.reshape(x, [self.batch_size, -1, 1])

        x = tf.concat([x,

                       historical_data,

                       np.expand_dims(np.tile(integral_constant, (self.output_timesteps,1)).T, axis=2),

                       np.expand_dims(np.tile(scale, (self.output_timesteps,1)).T, axis=2)

                      ], axis=2)

        x = BatchNormalization()(self.gru2(x))

        x = BatchNormalization()(self.gru2a(x))

        x = BatchNormalization()(self.gru_out(x))

        x = tf.reshape(x, [self.batch_size, -1])

        

        @tf.function

        def inverse_normalize(x):

            sales_pred = tf.transpose(tf.math.multiply(tf.transpose(x), y=scale))

            sales_pred = tf.math.cumsum(sales_pred, axis=1)

            sales_pred += np.tile(integral_constant, (self.output_timesteps,1)).T

            return sales_pred

        

        sales_pred = inverse_normalize(x)

        return sales_pred
from math import sqrt



IN_POINTS = 120

OUT_POINTS = 28

BATCH_SIZE = 16

model = M5_Net(input_timesteps=IN_POINTS, output_timesteps=OUT_POINTS, batch_size=BATCH_SIZE)



loss_object = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)



def loss(model, x, y, training):

    y_ = model(x, training=training)



    return loss_object(y_true=y, y_pred=y_)



def grad(model, inputs, targets):

    with tf.GradientTape() as tape:

        loss_value = loss(model, inputs, targets, training=True)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# model.load_weights('/kaggle/input/checkpoints2/croc_model2.ckpt')


M5_series_gen = M5_SeriesGenerator()

batch_sequence = [4, 4, 4, 4]

training = []

VAL_SIZE = 1000

validation = []

for epoch in range(len(batch_sequence)):

    BATCH_SIZE = batch_sequence[epoch]

    model.batch_size = BATCH_SIZE

    epoch_loss = []

    more_data_available, (X_train, y_train) = M5_series_gen.next_batch(in_points=IN_POINTS, out_points=OUT_POINTS, batch_size=BATCH_SIZE)

    while True:

        more_data_available, (X_train, y_train) = M5_series_gen.next_batch(in_points=IN_POINTS, out_points=OUT_POINTS, batch_size=BATCH_SIZE)

        if more_data_available == False:

            break;

            

        loss_value, grads = grad(model, X_train, y_train)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        clipped_loss = loss_object(y_true=y_train, y_pred=tf.clip_by_value(model(X_train, training=True), clip_value_min=0, clip_value_max=np.inf))

        epoch_loss.append(sqrt(clipped_loss))

    training.append(np.array(epoch_loss).mean())

    epoch_val = []

    for on, X_val in eval_series_data_gen(in_points=IN_POINTS, out_points=OUT_POINTS, end_of_data=1913, max_row=VAL_SIZE):

        model.batch_size = 1

        if on:

            val = tf.clip_by_value(model(X_val, training=True), clip_value_min=0, clip_value_max=np.inf).numpy().squeeze()

            epoch_val.append(val)

        else:

            epoch_val.append(np.zeros(OUT_POINTS))

        model.batch_size = BATCH_SIZE

    validation.append(np.array(epoch_val).mean())

    print(training[-1], validation[-1])

    print(f'Epoch {epoch} training loss: {training[-1]}, Epoch {epoch} validation loss: {validation[-1]}')

    model.save_weights('./croc_model{}.ckpt'.format(epoch))

    M5_series_gen.reset()
pd.Series(training).plot(legend=True, label='training')

pd.Series(validation).plot(legend=True, label='validation')
N = 16

model.batch_size = N

_, (X_train, y_train) = M5_series_gen.next_batch(in_points=IN_POINTS, out_points=OUT_POINTS, batch_size=N)

rows = np.arange(N)

fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10,N*3))

y_ = tf.clip_by_value(model(X_train, training=True), clip_value_min=0, clip_value_max=np.inf)

for ax, row in zip(axes, rows):

    pd.Series(y_train[row]).plot(legend=True, label='ground truth',ax=ax)

    pd.Series(y_[row]).plot(legend=True, label='forecast',ax=ax)

IN_POINTS = 120

OUT_POINTS = 28

model.batch_size = 1



validation = []

evaluation = []

iteration = 0

for  on, X in eval_series_data_gen(in_points=IN_POINTS, out_points=OUT_POINTS, end_of_data=1913):

    if on:

        val = tf.clip_by_value(model(X, training=True), clip_value_min=0, clip_value_max=np.inf).numpy().squeeze()

        validation.append(val)

    else:

        validation.append(np.zeros(OUT_POINTS))

        

for  on, X in eval_series_data_gen(in_points=IN_POINTS, out_points=OUT_POINTS, end_of_data=1941):

    if on:

        ev = tf.clip_by_value(model(X, training=True), clip_value_min=0, clip_value_max=np.inf).numpy().squeeze()

        evaluation.append(ev)

    else:

        evaluation.append(np.zeros(OUT_POINTS))



sample_submission.iloc[:30490, 1:] = validation

sample_submission.iloc[30490:, 1:] = evaluation

        

sample_submission.to_csv('./final_prediction.csv', index=False)

    