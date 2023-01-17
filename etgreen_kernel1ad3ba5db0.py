import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
data_dir = '/kaggle/input/covid19-global-forecasting-week-1'

pd.read_csv(f'{data_dir}/train.csv', parse_dates=['Date']).head()
COLUMN_NAME_MAP = {
    'Id': 'id',
    'ForecastId': 'id',
    'Province/State': 'province',
    'Country/Region': 'country',
    'Lat': 'lat',
    'Long': 'long',
    'Date': 'date',
    'ConfirmedCases': 'cases',
    'Fatalities': 'deaths'
}

def get_data(fn):
    """
    Consistent way to load and format both CSVs.
    """     
    df = (
        pd
        .read_csv(f'{data_dir}/{fn}.csv', parse_dates=['Date'])
        .rename(columns=COLUMN_NAME_MAP)
        .set_index('id')
        .replace(pd.np.nan, '')  # otherwise groupby ignores province = NaN
    )
    
    df['day'] = df.date.dt.dayofyear
    df['day'] -= 22 # first day in training set
    
    # convert these columns if training data
    if 'deaths' in df:
        df = df.astype({'deaths': int, 'cases': int})
    
    return df
    

train_df = get_data('train')

train_df.head()
cumulative_field_diff = (
    train_df
    .groupby(['country', 'province'])[['cases', 'deaths']]
    .diff()
    .fillna(0)
)

bad_rows = (cumulative_field_diff < 0).any(axis=1)
f'There are {bad_rows.sum()} bad rows where the cumulative number of cases or deaths decreases.'
train_df = train_df.loc[~bad_rows]
countries_train_df = train_df.groupby(['country', 'date', 'day'])['cases', 'deaths'].sum().reset_index()
most_cases_df = countries_train_df.groupby('country')['cases', 'deaths'].max().sort_values('cases', ascending=False)

most_cases_df.head(10)
n = 10
n_most_cases = most_cases_df.index.to_list()[:n]
g = sns.lineplot(x='date', y='cases', hue='country', data=countries_train_df.query('country in @n_most_cases'), estimator=None)
g.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=90);
g = sns.lineplot(x='date', y='deaths', hue='country', data=countries_train_df.query('country in @n_most_cases'), estimator=None)
g.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=90);
test_df = get_data('test')
x_predict = np.sort(test_df.day.unique())
x_all = np.arange(x_predict.max() + 1)
print(f'There are {len(x_predict)} to predict.')
print(f'Overlap of {train_df.day.max() - test_df.day.min()} days.')

test_df.head()
class ConstantModel:
    """
    A very simple model for when there is only one number of cases/deaths.
    
    Logisitic regression doesn't raises an exception in this case so use 
    this class instead.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y, sample_weight=None):
        self.value = y[0]
        
    def predict(self, X):
        return np.array([self.value] * X.shape[0])
    
    def __repr__(self):
        return f'ConstanModel(value={self.value})'


def run_models(grp_df, feature, target, x_to_predict, model, **model_kwargs):
            
    region_data = grp_df[[feature, target]].to_numpy()
    x = region_data[:, 0].reshape(-1, 1)
    y = region_data[:, 1]
    
    n_target_classes = np.unique(y).shape[0]
    model = model(**model_kwargs) if n_target_classes > 1 else ConstantModel()
    model.fit(x, y)
    
    result = {
        feature: x_predict,
        target: model.predict(x_to_predict.reshape(-1, 1)),
        f'{target}_model': model
    }
        
    return pd.DataFrame(result).set_index(feature)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression
model_kwargs = {
    'max_iter': 10_000
}

test_predictions_df = run_models(
    train_df.query('country == "Spain"'),
    feature='day',
    target='cases',
    x_to_predict=x_predict,
    model=LogisticRegression,
    **model_kwargs
)

display(test_predictions_df.head(10))
sns.lineplot(x='day', y='cases', data=test_predictions_df.reset_index());
region_columns = ['country', 'province']

death_predictions_df = (
    train_df
    .groupby(region_columns)
    .apply(run_models, feature='day', target='deaths', x_to_predict=x_predict, model=model, **model_kwargs)
    .reset_index()
)

cases_predictions_df = (
    train_df
    .groupby(region_columns)
    .apply(run_models, feature='day', target='cases', x_to_predict=x_predict, model=model, **model_kwargs)
    .reset_index()
)
merge_columns = ['country', 'province', 'day']
result_df = (
    test_df
    .reset_index()
    .merge(death_predictions_df, on=merge_columns)
    .merge(cases_predictions_df, on=merge_columns)
)

result_df
evaluation_df = train_df.merge(
    result_df, 
    on=merge_columns, 
    how='inner', 
    suffixes=('', '_predicted')
)[merge_columns + ['cases', 'cases_predicted', 'deaths', 'deaths_predicted']]


for model_type in ('cases', 'deaths',):
    formula = f'log({model_type}_predicted + 1) - log({model_type} + 1)'
    formula = f'({formula})**2'
    evaluation_df = evaluation_df.eval(f'{model_type}_rmsle = {formula}')
    
    
score_columns = ['cases_rmsle', 'deaths_rmsle']
(evaluation_df[score_columns].mean() ** 0.5).mean()
(
    evaluation_df
    .groupby(region_columns)[['cases_rmsle', 'deaths_rmsle']]
    .sum()
    .sort_values('cases_rmsle', ascending=False)
)
country_province = ('Estonia', '')
expr = 'country == @country_province[0] and province == @country_province[1]'
select_train_df = train_df.query(expr)
select_result_df = result_df.query(expr)

model_type = 'cases'
sns.scatterplot(x='day', y=model_type, data=select_train_df, label='actual');

predictions_all = (
    select_result_df[f'{model_type}_model']
    .iloc[0]
    .predict(x_all.reshape(-1, 1))
)
sns.scatterplot(x_all, predictions_all, label='prediction');
submission_df = (
    result_df
    .copy()
    .astype({'deaths': int, 'cases': int})
    .rename(columns={v: k for k, v in COLUMN_NAME_MAP.items()})
    .filter(['ForecastId', 'ConfirmedCases', 'Fatalities'])
)

# check that submission is in the correct format
expected_submission_df = pd.read_csv(f'{data_dir}/submission.csv')
for col in ('ConfirmedCases', 'Fatalities',):
    expected_submission_df[col] = submission_df[col]  
pd.testing.assert_frame_equal(submission_df, expected_submission_df)

submission_df.to_csv('submission.csv', index=False)
for model_type in ('cases', 'deaths',):
    train_df[f'daily_{model_type}'] = train_df.groupby(region_columns)[model_type].diff().fillna(0).astype(int)
    train_df.query('country == "United Kingdom" and province == "United Kingdom"').plot(x='day', y=f'daily_{model_type}')
for window in (3, 5, 10):
    train_df[[f'daily_cases_win_{window}', f'daily_deaths_win_{window}']] = (
        train_df[['daily_cases', 'daily_deaths']].rolling(window, min_periods=1).mean()
    )
    
train_df.head()
for model_type in ('cases', 'deaths',):
    train_df.query('country == "United Kingdom" and province == "United Kingdom"').plot(x='day', y=f'daily_{model_type}_win_10')
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
def univariate_data(df, start_index, end_index, history_size, target_size):
    """
    abcdef -> [abc, bcd, cde], [d, e, f]
    for history_size = 3, target_size = 1
    """
    data = []
    labels = []
    
    dataset = df.to_numpy()
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    if end_index <= start_index:
        raise ValueError(f'End index {end_index} not greater than start index {start_index}.')

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(dataset[indices].reshape(history_size, 1))
        labels.append(dataset[i+target_size])

    return data, labels


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step=1):
    data = []
    labels = []
    
    dataset = dataset.to_numpy()
    target = target.to_numpy()

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    if end_index <= start_index:
        raise ValueError(f'End index {end_index} not greater than start index {start_index}.')

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices].reshape(history_size, -1))

        if step > 1:
            labels.append(target[i:i+target_size])
        else:
            labels.append(target[i+target_size])

    return data, labels
history = 5
forecast_jump = 3
features = 'day'
scale = True

train_sample_size = test_df['day'].min()
scaled_df = train_df.copy()

if scale:
    scaled_df[features] = (scaled_df[features] - scaled_df[features].mean()) / scaled_df[features].std()

xs = []
ys = []
regions = []

for grp_name, grp_df in scaled_df.groupby(['country', 'province']):
    grp_x, grp_y = univariate_data(grp_df[features], 0, train_sample_size, history, forecast_jump)
#     grp_x, grp_y = multivariate_data(
#         grp_df[features], grp_df[features], 0, train_sample_size, history, forecast_jump
#     )
    xs += grp_x
    ys += grp_y
    regions += [grp_name] * len(grp_y)
    break
    
x_train_uni = np.array(xs)
y_train_uni = np.array(ys)
regions_train = np.array(regions)
x_train_uni.shape
y_train_uni.shape
print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])
scaled_df = train_df.copy()

if scale:
    scaled_df[features] = (scaled_df[features] - scaled_df[features].mean()) / scaled_df[features].std()

xs = []
ys = []
regions = []

for grp_name, grp_df in scaled_df.groupby(['country', 'province']):
    # swapped the 2nd and 3rd args below
    #grp_x, grp_y = univariate_data(grp_df[features], train_sample_size, None, history, forecast_jump)
    grp_x, grp_y = multivariate_data(
        grp_df[features], grp_df[features], train_sample_size, None, history, forecast_jump
    )
    xs += grp_x
    ys += grp_y
    regions += [grp_name] * len(grp_y)
    
x_val_uni = np.array(xs)
y_val_uni = np.array(ys)
regions_val = np.array(regions)
i = 0
print ('Single window of past history')
print (x_val_uni[i])
print ('\n Target temperature to predict')
print (y_val_uni[i])
def create_time_steps(length):
    return list(range(-length, 0))


def baseline(history):
    return np.mean(history)


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    future = delta if delta else 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    
    return plt
show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example');
show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example');
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
EVALUATION_INTERVAL = 200
EPOCHS = 10

single_step_history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50);
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    

plot_train_history(single_step_history, 'Single Step Training and validation loss')
for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()