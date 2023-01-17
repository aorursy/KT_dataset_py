# data manipulation

import numpy as np

import pandas as pd



# for plotting purposes

import seaborn as sns

import matplotlib.pyplot as plt



# options

pd.options.display.max_columns = 300



import warnings

warnings.filterwarnings("ignore")





df = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRES_PVGIS_TSh_CF_n2_19862015.csv")





def add_date_time(_df):

    "Returns a DF with two new cols : the time and hour of the day"

    t = pd.date_range(start='1/1/1986', periods=_df.shape[0], freq = 'H')

    t = pd.DataFrame(t)

    _df = pd.concat([_df, t], axis=1)

    _df.rename(columns={ _df.columns[-1]: "time" }, inplace = True)

    _df['year'] = _df['time'].dt.year

    _df['month'] = _df['time'].dt.month

    _df['week'] = _df['time'].dt.weekofyear

    _df['day'] = _df['time'].dt.dayofyear    

    _df['hour'] = _df['time'].dt.hour

    return _df
df = add_date_time(df)

df = df[~df.year.isin([1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016])]



# keeping only values for one country for the predictions

df = df[['FR10', 'year', 'month', 'week', 'day', 'hour', 'time']]

df.head(2)
df.shape
# train data including 10 years of records except the last month

train_data = df[-24*365*10:-24*31]



# test data = last month of records for year 2015

test_data = df[-24*31:]
from sklearn.metrics import mean_squared_error

model_instances, model_names, rmse_train, rmse_test = [], [], [], []





def plot_scores():

    """Create three lists : models, the RMSE on the train set and on the test set, then plot them"""

    df_score = pd.DataFrame({'model_names' : model_names,

                             'rmse_train' : rmse_train,

                             'rmse_test' : rmse_test})

    df_score = pd.melt(df_score, id_vars=['model_names'], value_vars=['rmse_train', 'rmse_test'])



    plt.figure(figsize=(12, 10))

    sns.barplot(y="model_names", x="value", hue="variable", data=df_score)

    plt.show()
x_train, y_train = train_data.drop(columns=['time']), train_data['FR10']

x_test, y_test = test_data.drop(columns=['time']), test_data['FR10']





def mean_df(d, h):

    "return the hourly mean of a specific day of the year"

    res = x_train[(x_train['day'] == d) & (x_train['hour'] == h)]['FR10'].mean()

    return res



    # examples 

    #df['col_3'] = df.apply(lambda x: f(x.col_1, x.col_2), axis=1)

    # x_train[(x_train['day'] == x['day']) & (x_train['hour'] == x['hour'])]['FR10'].mean()

    

    

#x_train['pred'] = x_train.apply(lambda x: mean_df(x.day, x.hour), axis=1)

x_test['pred'] = x_test.apply(lambda x: mean_df(x.day, x.hour), axis=1)

x_test.head()
model_names.append("base_line")

rmse_train.append(np.sqrt(mean_squared_error(x_train['FR10'], x_train['FR10']))) # a modifier en pred

rmse_test.append(np.sqrt(mean_squared_error(x_test['FR10'], x_test['pred'])))
def plot_predictions(data):

    plt.figure(figsize=(14, 6))

    sns.lineplot(data=data)

    plt.title("Base line predictions (orange) vs real values (blue) for the last month")

    plt.xlabel("hours of the last month (12-2015)")

    plt.ylabel("solar installation efficiency")

    plt.show()



plot_predictions(data=x_test[['FR10', 'pred']])
X_train, y_train = train_data[['month', 'week', 'day', 'hour']], train_data['FR10']

X_test, y_test = test_data[['month', 'week', 'day', 'hour']], test_data['FR10']

X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import LinearSVR

from sklearn.svm import SVR



import xgboost as xgb

import lightgbm as lgbm





def get_rmse(reg, model_name):

    """Print the score for the model passed in argument and retrun scores for the train/test sets"""

    

    y_train_pred, y_pred = reg.predict(X_train), reg.predict(X_test)

    rmse_train, rmse_test = np.sqrt(mean_squared_error(y_train, y_train_pred)), np.sqrt(mean_squared_error(y_test, y_pred))

    print(model_name, f'\t - RMSE on Training  = {rmse_train:.2f} / RMSE on Test = {rmse_test:.2f}')

    

    return rmse_train, rmse_test





# list of all the basic models used at first

model_list = [

    LinearRegression(), Lasso(), Ridge(), ElasticNet(),

    RandomForestRegressor(), GradientBoostingRegressor(), ExtraTreesRegressor(),

    xgb.XGBRegressor(), lgbm.LGBMRegressor(), KNeighborsRegressor()

             ]



# creation of list of names and scores for the train / test

model_names.extend([str(m)[:str(m).index('(')] for m in model_list])





# fit and predict all models

for model, name in zip(model_list, model_names):

    model.fit(X_train, y_train)

    sc_train, sc_test = get_rmse(model, name)

    rmse_train.append(sc_train)

    rmse_test.append(sc_test)
print("""

svm_lin = LinearSVR()

svm_lin.fit(X_train, y_train)

sc_train, sc_test = get_rmse(svm_lin, "SVM lin.")

model_names.append("SVM lin.")

rmse_train.append(sc_train)

rmse_test.append(sc_test)



SVM lin. 	 - RMSE on Training  = 0.31 / RMSE on Test = 0.30

""")
print("""

svm_poly = SVR(kernel='poly', degree=4, max_iter=100)

svm_poly.fit(X_train, y_train)

sc_train, sc_test = get_rmse(svm_poly, "SVM poly.")

model_names.append("SVM poly.")

rmse_train.append(sc_train)

rmse_test.append(sc_test)



SVM poly. 	 - RMSE on Training  = 0.52 / RMSE on Test = 0.56

""")
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline





poly_lin_reg = Pipeline([

    ("poly_feat", PolynomialFeatures(degree=4)),

    ("linear_reg", LinearRegression())

])



poly_lin_reg.fit(X_train, y_train)



sc_train, sc_test = get_rmse(poly_lin_reg, "Poly Linear Reg")



model_names.append('Poly Linear Reg')

rmse_train.append(sc_train)

rmse_test.append(sc_test)
# train data for 10 years

train_data_d = df[-24*365*10:][['FR10', 'month', 'week', 'day', 'hour']]



# one hot encoding for categorical feature

cat_feat = ['month', 'week', 'day', 'hour']

train_data_d = pd.get_dummies(data=train_data_d, columns=cat_feat, drop_first=True)

train_data_d.head()



# keep last month for the test data set

test_data_d, train_data_d = train_data_d[-24*31:], train_data_d[:-24*31]



# get_dummies or one hot encoding

X_train_d, y_train_d = train_data_d.drop(columns=['FR10']), train_data_d['FR10']

X_test_d, y_test_d = test_data_d.drop(columns=['FR10']), test_data_d['FR10']



# verify if different shapes match

X_train_d.shape, y_train_d.shape, X_test_d.shape, y_test_d.shape
categorical_linreg = LinearRegression()

categorical_linreg.fit(X_train, y_train)

sc_train, sc_test = get_rmse(categorical_linreg, "Categorical lin. reg.")

print("Not more efficient than linear regression without get dummies")
gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)

y_pred = pd.DataFrame(gbr.predict(X_test))



y_test = pd.DataFrame(y_test)

y_test['pred'] = y_pred.values



plot_predictions(data=y_test)
# reload the data

df = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRES_PVGIS_TSh_CF_n2_19862015.csv")

df = add_date_time(df)

#df = df[~df.year.isin([1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016])]





# only keep what it usefull to use here

data = df[['time', 'FR10']]

data = data.rename(columns={"time": "ds", "FR10": "y"})



# train set : 10 yrs. except last month

train_data = data[-24*365*10:-24*31]



# test set = last month of the record (2015)

test_data = data[-24*31:]
from fbprophet import Prophet



prophet_model = Prophet()

prophet_model.fit(train_data)

test_data.tail()

#y_train_pred = prophet_model.predict(train_data)
forecast = prophet_model.predict(test_data)

forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forecast['FR10'] = test_data['y'].values

forecast.tail()
model_names.append("prophet") #.extend(["prophet", "prophet_lower", "prophet_upper"])



rmse_train.append(0)

#rmse_train.extend([0, 0, 0])

rmse_test.append(np.sqrt(mean_squared_error(test_data['y'], forecast['yhat'])))

#rmse_test.append(np.sqrt(mean_squared_error(test_data['y'], forecast['yhat_lower'])))

#rmse_test.append(np.sqrt(mean_squared_error(test_data['y'], forecast['yhat_upper'])))





plot_predictions(data=forecast[['FR10', 'yhat']])



# if we wanted to plot aslo inf / sup

# plt.figure(figsize=(18, 8))

# sns.lineplot(data=forecast[['FR10', 'yhat_lower', 'yhat_upper']])
df = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRES_PVGIS_TSh_CF_n2_19862015.csv")

df = df[sorted([c for c in df.columns if 'FR' in c])]



# keep only 4 years

df = df[-24*365*4:]



# nb lines / cols

df.shape
def process_data(data, past):

    X = []

    for i in range(len(data)-past-1):

        X.append(data.iloc[i:i+past].values)

    return np.array(X)





lookback = 48



y = df['FR10'][lookback+1:] 

X = process_data(df, lookback)

X.shape, y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Dropout





def my_RNN():

    my_rnn = Sequential()

    my_rnn.add(SimpleRNN(units=32, return_sequences=True, input_shape=(lookback,22)))

    my_rnn.add(SimpleRNN(units=32, return_sequences=True))

    my_rnn.add(SimpleRNN(units=32, return_sequences=False))

    my_rnn.add(Dense(units=1, activation='linear'))

    return my_rnn





rnn_model = my_RNN()

rnn_model.compile(optimizer='adam', loss='mean_squared_error')

rnn_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
y_pred_train, y_pred_test = rnn_model.predict(X_train), rnn_model.predict(X_test)

err_train, err_test = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))

err_train, err_test
def append_results(model_name):

    model_names.append(model_name)

    rmse_train.append(err_train)

    rmse_test.append(err_test)



append_results("RNN")





def plot_evolution():

    plt.figure(figsize=(12, 6))

    plt.plot(np.arange(len(X_train)), y_train, label='Train')

    plt.plot(np.arange(len(X_train), len(X_train)+len(X_test), 1), y_test, label='Test')

    plt.plot(np.arange(len(X_train), len(X_train)+len(X_test), 1), y_pred_test, label='Test prediction')

    plt.legend()

    plt.show()



plot_evolution()
rnn_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])

plot_predictions(data=rnn_res[-30*24:])
from tensorflow.keras.layers import GRU



def my_GRU(input_shape):

    my_GRU = Sequential()

    my_GRU.add(GRU(units=32, return_sequences=True, activation='relu', input_shape=input_shape))

    my_GRU.add(GRU(units=32, activation='relu', return_sequences=False))

    my_GRU.add(Dense(units=1, activation='linear'))

    return my_GRU



gru_model = my_GRU(X.shape[1:])

gru_model.compile(optimizer='adam', loss='mean_squared_error')

gru_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
y_pred_train, y_pred_test = gru_model.predict(X_train), gru_model.predict(X_test)

err_train, err_test = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))

err_train, err_test
append_results("GRU")

plot_evolution()



gru_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])

plot_predictions(data=gru_res[-30*24:])
from tensorflow.keras.layers import LSTM



def my_LSTM(input_shape):

    my_LSTM = Sequential()

    my_LSTM.add(LSTM(units=32, return_sequences=True, activation='relu', input_shape=input_shape))

    my_LSTM.add(LSTM(units=32, activation='relu', return_sequences=False))

    my_LSTM.add(Dense(units=1, activation='linear'))

    return my_LSTM



lstm_model = my_LSTM(X.shape[1:])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
y_pred_train, y_pred_test = lstm_model.predict(X_train), lstm_model.predict(X_test)

err_train, err_test = np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))

err_train, err_test
append_results("LSTM")

plot_evolution()



lstm_res = pd.DataFrame(zip(list(y_test), list(np.squeeze(y_pred_test))), columns =['FR10', 'pred'])

plot_predictions(data=lstm_res[-30*24:])
#len(model_names), len(rmse_train), len(rmse_test)



plt.style.use('fivethirtyeight')

plot_scores()