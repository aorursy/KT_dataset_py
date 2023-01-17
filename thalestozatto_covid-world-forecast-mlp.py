import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

import datetime

import sklearn

from sklearn.preprocessing import MinMaxScaler

import keras

from keras.layers import Dense, ELU, Activation, Layer, Dropout

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks.callbacks import ModelCheckpoint

from keras import backend as K

from keras.losses import mean_squared_logarithmic_error as msle_keras

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_log_error as msle

from sklearn import preprocessing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

        print(file_path)

        if filename == 'train.csv':

            df_train_full = pd.read_csv(file_path, index_col=0).fillna('')

        elif filename == 'test.csv':

            df_test = pd.read_csv(file_path, index_col=0).fillna('')

        elif filename == 'submission.csv':

            df_sub = pd.read_csv(file_path, index_col=0)
df_test.head()
first_dates = {}

r_c = {}

r_f = {}

for country in pd.unique(df_train_full['Country/Region']):

    df_aux = df_train_full.loc[df_train_full['Country/Region'] == country]

    for state in pd.unique(df_aux['Province/State']):

        find_date = True

        df_aux_ = df_aux.loc[df_aux['Province/State'] == state]

        r_list = [[], []]

        for row in df_aux_.iterrows():

            if row[0] > 1:

                if prev_row[1]['ConfirmedCases'] > 0:

#                     r_list[0].append(row[1]['ConfirmedCases']/prev_row[1]['ConfirmedCases'])

                    r_c[(country, state, row[1]['Date'])] = row[1]['ConfirmedCases']/prev_row[1]['ConfirmedCases']

                if prev_row[1]['Fatalities'] > 0:

#                     r_list[1].append(row[1]['Fatalities']/prev_row[1]['Fatalities'])

                    r_f[(country, state, row[1]['Date'])] = row[1]['Fatalities']/prev_row[1]['Fatalities']

            if row[0] == 1 or prev_row[1]['ConfirmedCases'] == 0:

                r_c[(country, state, row[1]['Date'])] = 1

            if row[0] == 1 or prev_row[1]['Fatalities'] == 0:

                r_f[(country, state, row[1]['Date'])] = 1

            prev_row = row

            

            if row[1]['ConfirmedCases'] == 0 :

                first_dates[(country, state)] = ''

            elif row[1]['ConfirmedCases'] > 0 and find_date:

                first_dates[(country, state)] = row[1]['Date']

#                 print('{} - {}: {}'.format(country, state, first_dates[(country, state)]))

                find_date = False



#         print(r_list)

#         r_c[(country, state)] = np.mean(r_list[0])

#         r_f[(country, state)] = np.mean(r_list[1])
def add_first_date(df, first_dates):

    for row in df.iterrows():

        country = row[1]['Country/Region']

        state = row[1]['Province/State']

        df.loc[row[0], 'FirstCase'] = first_dates[(country, state)]

    return df
def add_r(df, r_c, r_f):

    df['r_c'] = 1

    df['r_f'] = 1

    for row in df.iterrows():

        country = row[1]['Country/Region']

        state = row[1]['Province/State']

        date = row[1]['Date']

        df.loc[row[0], 'r_c'] = r_c[(country, state, date)]

        df.loc[row[0], 'r_f'] = r_f[(country, state, date)]

    return df
df_train_fd = add_first_date(df_train_full, first_dates)

# df_train_full = add_r(df_train_full, r_c, r_f).fillna(1)

df_test_fd = add_first_date(df_test, first_dates)

# df_test = add_r(df_test, r_c, r_f).fillna(1)
df_eval = df_train_fd[df_train_fd["Date"] >= min(df_test_fd['Date'])]

df_train = df_train_fd[df_train_fd["Date"] < min(df_test_fd['Date'])]

df_eval
countries = pd.unique(df_train['Country/Region'])

print("Number of countries: ", len(countries))

days = pd.unique(df_train['Date'])

print("Number of days: ", len(days))
def groupby_country(df, date):

    df_aux = df.loc[df['Date'] == date]

    df_r = pd.DataFrame(columns=['country','confirmed', 'deaths'])

    if 'pred_c' in df.columns:

        df_r = pd.DataFrame(columns=['country','confirmed', 'deaths', 'pred_c', 'pred_f'])

    countries = pd.unique(df['Country/Region'])

    df_r['country'] = countries

    df_r.set_index('country', inplace=True)

    for c in countries:

        df_r.loc[c]['confirmed'] = int(sum(df_aux.loc[df_aux['Country/Region'] == c]['ConfirmedCases']))

        df_r.loc[c]['deaths'] = int(sum(df_aux.loc[df_aux['Country/Region'] == c]['Fatalities']))

        if 'pred_c' in df.columns:

            df_r.loc[c]['pred_c'] = int(sum(df_aux.loc[df_aux['Country/Region'] == c]['pred_c']))

            df_r.loc[c]['pred_f'] = int(sum(df_aux.loc[df_aux['Country/Region'] == c]['pred_f']))

    return df_r
df_c = groupby_country(df_eval, '2020-03-23')

fig = px.choropleth(df_c, locations=df_c.index, 

                    locationmode='country names', 

                    color = df_c['confirmed'].astype(int),

                    hover_name=df_c.index,

                    color_continuous_scale="peach", 

                    title='Countries with Confirmed Cases until 2020-03-22',

                    range_color=(0,5000),

                    labels = {'color':'Confirmed Cases'},

                   )

fig.show()
def plot_country(df, countries, features, log=True):

    """

    countries and features must be lists

    """



    fig = go.Figure({'layout':{'title':{'text':"COVID-19 cases by country"}}})

    for country in countries:

        df_aux = df.loc[df['Country/Region'] == country]

        days = pd.unique(df_aux['Date'])

        for feature in features:

            s = []

            for day in days:

                s.append(int(groupby_country(df_aux, day)[feature]))

            fig.add_trace(go.Scatter(x=days, y=s, mode='lines', name='{} in {}'.format(feature, country)))

    fig.update_xaxes(title='Date')

    fig.update_yaxes(title='cases')

    if log:

        fig.update_yaxes(type=('log'))

        

    fig.show()
plot_country(df_train, ['Brazil', 'China', 'Italy', 'Iran'], ['confirmed'])
def df2array(df, le, test=False):   

#     min_date = datetime.datetime.strptime(df["Date"].min(), "%Y-%m-%d") + datetime.timedelta(3)

#     df_aux = df.loc[df["Date"] > min_date.strftime("%Y-%m-%d")]

    

    x = np.zeros((len(df), 5))

    y = np.zeros((len(df), 2))

    

    for i in range(len(x)):

        dtm = datetime.datetime.strptime(df.iloc[i]['Date'], "%Y-%m-%d")

        if df.iloc[i]['FirstCase'] == '':

            x[i, 4] = -1

        else:

            dtm_f = datetime.datetime.strptime(df.iloc[i]['FirstCase'], "%Y-%m-%d")

            x[i, 4] = datetime.datetime.timestamp(dtm_f)

#         dtm_1 = dtm - datetime.timedelta(1)

#         dtm_2 = dtm - datetime.timedelta(2)

#         dtm_3 = dtm - datetime.timedelta(3)

        

        lat = df.iloc[i]['Lat']

        long = df.iloc[i]['Long']

        location = le.transform(['{} - {}'.format(df.iloc[i]['Country/Region'], df.iloc[i]['Province/State'])])

        first_date = df.iloc[i]['FirstCase']

#         r_c = df.iloc[i]['r_c']

#         r_f = df.iloc[i]['r_f']

        

#         df_aux_ = df.loc[(df["Country/Region"] == country) &

#                         (df["Province/State"] == state)]

                

        x[i, 0] = datetime.datetime.timestamp(dtm)

        x[i, 1] = lat

        x[i, 2] = long

        x[i, 3] = location

#         x[i, 5] = r_c

#         x[i, 6] = r_f

        

        

#         x[i, 5] = df_aux_.loc[df_aux_["Date"] == dtm_1.strftime("%Y-%m-%d")]["ConfirmedCases"]

#         x[i, 6] = df_aux_.loc[df_aux_["Date"] == dtm_2.strftime("%Y-%m-%d")]["ConfirmedCases"]

#         x[i, 7] = df_aux_.loc[df_aux_["Date"] == dtm_3.strftime("%Y-%m-%d")]["ConfirmedCases"]

#         x[i, 8] = df_aux_.loc[df_aux_["Date"] == dtm_1.strftime("%Y-%m-%d")]["Fatalities"]

#         x[i, 9] = df_aux_.loc[df_aux_["Date"] == dtm_2.strftime("%Y-%m-%d")]["Fatalities"]

#         x[i, 10] = df_aux_.loc[df_aux_["Date"] == dtm_3.strftime("%Y-%m-%d")]["Fatalities"]



        if not test:

            y[i, 0] = df.iloc[i]['ConfirmedCases']

            y[i, 1] = df.iloc[i]['Fatalities']

            

    if not test:

        return x, y

    else:

        return x
df_merg = df_test.merge(df_eval, on=['Country/Region', 'Province/State', 'Date', 'Lat', 'Long', 'FirstCase'], right_index=True)

df_merg
locations = []

for row in df_train.iterrows():

    locations.append('{} - {}'.format(row[1]['Country/Region'], row[1]['Province/State']))



le = preprocessing.LabelEncoder()

le.fit(locations)
X, y = df2array(df_train, le)

X_eval, y_eval = df2array(df_merg, le)

X_test = df2array(df_test, le, test=True)



X_all = np.vstack((X, X_test))

scaler = MinMaxScaler()

scaler.fit(X_all)



X = scaler.transform(X)

X_eval = scaler.transform(X_eval)

X_test = scaler.transform(X_test)
# kf = KFold(n_splits=4, random_state=36, shuffle=True)

# kf.get_n_splits(X)

# i=0

# loss = []

# history = []



# for train_index, val_index in kf.split(X):

#     print("K", i+1)

#     X_train, X_val = X[train_index], X[val_index]

#     y_train, y_val = y[train_index], y[val_index]

    

#     model = build_model()

#     model.compile(loss=RMSLE_keras, optimizer='adam')

#     history.append(model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), 

#               shuffle=False, epochs=200, batch_size=64))

#     rmsle = RMSLE(y_val, model.predict(X_val))

#     loss.append(rmsle)

#     print("{} - RMSLE: {}".format(i, rmsle))

#     i+=1

    

# print("mean: {} std: {}".format(np.mean(loss), np.std(loss)))
# fig = go.Figure({'layout':{'title':{'text':"Train and Validation Loss"}}})



# fig.add_trace(go.Scatter(y=history[0].history['loss'], mode='lines', name='Train loss'))

# fig.add_trace(go.Scatter(y=history[0].history['val_loss'], mode='lines', name='Validation loss'))

# fig.update_xaxes(title='Epoch')

# fig.update_yaxes(title='RMSLE')

# fig.show()
def build_model(input_shape):

    model = Sequential()

    model.add(Dense(256, input_shape=(input_shape,)))

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(128))

    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(64))

    model.add(Activation('relu'))

    model.add(Dense(32))

    model.add(Activation('relu'))

    model.add(Dense(2))

    model.add(Activation('relu'))

    

    return model
model = build_model(X.shape[1])

opt = Adam(0.001)

model.compile(loss=msle_keras, optimizer=opt)

ckpt = ModelCheckpoint('ckpt', save_best_only=True)

history = model.fit(x=X, y=y, validation_data=(X_eval, y_eval), epochs=500, batch_size=128, callbacks=[ckpt])

model.load_weights('ckpt')
fig = go.Figure(go.Scatter(y=history.history['loss'], mode='lines', name='Loss'))

fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name="Val Loss"))

fig.show()
preds_eval = model.predict(X_eval)



df_eval.loc[:, "pred_c"] = preds_eval[:, 0]

df_eval.loc[:, "pred_f"] = preds_eval[:, 1]

df_eval
score_c = np.sqrt(msle(y_eval[:, 0], preds_eval[:, 0]))

score_f = np.sqrt(msle(y_eval[:, 1], preds_eval[:, 1]))



print('score_c: {}, score_f: {}, mean: {}'.format(score_c, score_f, np.mean([score_c, score_f])))
df_test.head(5)
preds_sub = model.predict(X_test)

df_sub.loc[:] = preds_sub

df_sub
df_sub.to_csv('submission.csv')
df_eval.loc[df_eval["Country/Region"] == "Brazil"]
plot_country(df_train, ["Brazil"], ["confirmed"], log=False)
plot_country(df_eval, ["Brazil"], ["confirmed", "pred_c"], log=False)