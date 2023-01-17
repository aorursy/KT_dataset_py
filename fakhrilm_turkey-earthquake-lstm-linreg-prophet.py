# Install Facebook's Prophet library

!pip install fbprophet
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt
ROOT = '../input/earthquake/'
# Read file

df = pd.read_csv(ROOT + 'earthquake.csv')

print (df.shape)

df.head(5)
import datetime



# Extract year for filtering purpose

year = []

for index, row in df.iterrows():

    try:

        date = row['date']

        date_time_obj = datetime.datetime.strptime(date, '%Y.%d.%m')

        y = date_time_obj.date().year

        year.append(y)

    except:

        year.append(-1)

print (year[:5])
# Insert new column 'Year'

df.insert(loc=1, column='Year', value=year)
df.shape
# Filtering data with recent (last 27 year) earthquake

sample = df.loc[df['Year'] >= 1970]

print(sample.shape)

sample.head(5)
# Earthquake directions

directions = list(set(df['direction']))

directions = [x for x in directions if str(x) != 'nan']
import folium

from IPython.display import display



# create the map.

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'grey', 'black', 'purple']

map_pickup = folium.Map( location=[39.011042, 35.250453], zoom_start=6)

count = {}

for i in range(8):

    d = directions[i]

    c = colors[i]

    f = sample[sample['direction'] == d]

    count[d] = len(f)

    f.apply(lambda row : folium.CircleMarker(location=[row["lat"], row["long"]], color=c).add_to(map_pickup), axis=1)
display(map_pickup)
plt.figure(figsize=(10,4))

plt.bar(range(len(count)), list(count.values()), align='center')

plt.xticks(range(len(count)), list(count.keys()))

plt.show()
frame = pd.DataFrame(columns=directions)

for year in range(1910, 2018):

    temp = []

    # ['east', 'north', 'south_west', 'north_west', 'south_east', 'south', 'west', 'north_east']

    for d in directions:

        f = df[(df['direction'] == d) & (df['Year'] == year)]

        temp.append(len(f))

    frame.loc[year] = temp
frame.plot.line(figsize=(10,6)).grid(b=True, which='both')
frame[frame.index.isin([i for i in range(1960,1971)])].plot.line()
filtered_frame = frame[frame.index.isin([i for i in range(1970,2018)])].astype('float')

filtered_frame.head(5)
import seaborn as sn



corr = filtered_frame.corr()

fig, ax = plt.subplots(figsize=(12,12))

sn.heatmap(corr, annot=True, ax=ax)
corr = filtered_frame.filter(items=['south_west', 'north_west', 'south_east', 'north_east']).corr()

fig, ax = plt.subplots(figsize=(8,8))

sn.heatmap(corr, annot=True, ax=ax)
train_frame = filtered_frame.filter(items=['south_west', 'north_west', 'south_east', 'north_east'])

train_frame.head(5)
from sklearn.ensemble import RandomForestRegressor
models = {}

train = train_frame[train_frame.index < 2000]

test = train_frame[train_frame.index >= 2000]



print('Random Forest - R2 Scores : ')

for col in list(train_frame.columns):

    train_cols = list(set(list(train_frame.columns)) - set([col]))

    

    x_train = np.array(train.filter(items=train_cols))

    x_test = np.array(test.filter(items=train_cols))

    y_train = np.array(train[col])

    y_test = np.array(test[col])

    

    m = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)

    m.fit(x_train, y_train)

    models[col] = m

    print(col, ' : ', m.score(x_test, y_test))
from sklearn.linear_model import LinearRegression
print('Linear Regression - R2 Scores : ')

lr = {}

for col in list(train_frame.columns):

    train_cols = list(set(list(train_frame.columns)) - set([col]))

    

    x_train = np.array(train.filter(items=train_cols))

    x_test = np.array(test.filter(items=train_cols))

    y_train = np.array(train[col])

    y_test = np.array(test[col])

    

    m = LinearRegression()

    m.fit(x_train, y_train)

    lr[col] = {'m' : m, 'y_true' : y_test, 'y_pred' : np.array(m.predict(x_test)) }

    print(col, ' : ', m.score(x_test, y_test))
frame
fig, ax = plt.subplots(2, 2, figsize=(15,10))

keys = [i for i in range(2001, 2017)]

row = 0

col = 0

for key in lr.keys():

    y_true = lr[key]['y_true']

    y_pred = lr[key]['y_pred']

    

    ax[row, col].plot(y_true, label='True')

    ax[row, col].plot(y_pred, label='Prediction')

    

    ax[row, col].legend(loc="upper left")

    ax[row, col].set_xticklabels(keys)

    ax[row, col].set_title(key)

    

    if col == 1:

        row += 1

        col = 0

    else:

        col += 1
frame
x = [[i] for i in range(1910, 2000)]

models_year = {}

for d in directions:

    y = np.array(frame[d].loc[frame.index < 2000])

    m = LinearRegression()

    m.fit(x, y)

    models_year[d] = {'m' : m}
x = [[i] for i in range(2000, 2018)]

for d in directions:

    y_true = np.array(frame[d].loc[frame.index >= 2000])

    y_pred = models_year[d]['m'].predict(x)
from fbprophet import Prophet

from fbprophet.plot import plot
main_frame = df[df['Year'] >= 1970]

annual_frame = main_frame.groupby(['Year']).count()

annual_frame = annual_frame[['id']]

annual_frame = annual_frame.reset_index(level=0)

annual_frame = annual_frame.rename({'Year' : 'ds', 'id' : 'y'}, axis=1)

annual_frame['ds'] = annual_frame.apply(lambda row : datetime.datetime.strptime(str(row.ds) + '-12-31', '%Y-%m-%d'), axis=1)

annual_frame[annual_frame.apply(lambda row : int(str(row['ds']).split('-')[0]) <= 2000, axis=1)]

annual_train = annual_frame[annual_frame.apply(lambda row : int(str(row['ds']).split('-')[0]) <= 2000, axis=1)]

annual_test = annual_frame[annual_frame.apply(lambda row : int(str(row['ds']).split('-')[0]) > 2000, axis=1)]

annual_test
annual_model = Prophet()

annual_model.fit(annual_train)
future = annual_model.make_future_dataframe(periods=17, freq='A', include_history=True)

forecast = annual_model.predict(future)
fig = annual_model.plot(forecast)
y_true = list(annual_test['y'].values)

y_mean = sum(y_true)/len(y_true)

y_mean = [y_mean for i in range(len(y_true))]

y_pred = list(forecast.tail(17)['yhat'].values)

y_pred = [0 if i < 0 else i for i in y_pred]
plt.figure(figsize=(10,5))

plt.plot(y_true, label='True')

plt.plot(y_pred, label='Pred')

plt.plot(y_mean, label='Baseline (Mean)')

plt.legend(loc="upper left")

plt.grid(True)
r2 = r2_score(y_true, y_pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))

print(r2, rmse)
nn_train = list(annual_train.y.values)

nn_test = list(annual_test.y.values)
def extract_features(arr):

    x_arr = []

    y_arr = arr[3:]

    for i in range(3,len(arr)): 

        x_arr.append(arr[i-3:i])

    return x_arr, y_arr
nn_train_x, nn_train_y = extract_features(nn_train)

nn_test_x, nn_test_y = extract_features(nn_test)

mlp = make_pipeline(

       MLPRegressor(

           hidden_layer_sizes=(100, 100),

           tol=1e-2, max_iter=500, random_state=0,

           solver='adam'

       ))

mlp.fit(X=nn_train_x, y=nn_train_y)
y_true = np.array(nn_test_y)

y_pred = np.array(mlp.predict(nn_test_x))

y_mean = sum(y_true)/len(y_true)

y_mean = [y_mean for i in range(len(y_true))]
plt.figure(figsize=(10,5))

plt.plot(y_true, label='True')

plt.plot(y_pred, label='Pred')

plt.plot(y_mean, label='Baseline (Mean)')

plt.legend(loc="upper left")

plt.grid(True)
r2 = r2_score(y_true, y_pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))

print(r2, rmse)
lgr = make_pipeline(LinearRegression())

lgr.fit(X=nn_train_x, y=nn_train_y)
y_true = np.array(nn_test_y)

y_pred = np.array(lgr.predict(nn_test_x))

y_mean = sum(y_true)/len(y_true)

y_mean = [y_mean for i in range(len(y_true))]
plt.figure(figsize=(10,5))

plt.plot(y_true, label='True')

plt.plot(y_pred, label='Pred')

plt.plot(y_mean, label='Baseline (Mean)')

plt.legend(loc="upper left")

plt.grid(True)
r2 = r2_score(y_true, y_pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))

print(r2, rmse)
class TemporaryClass:

    def __init__(self, direction):

        self.d = direction

        prophet_data = filtered_frame.filter(items=[self.d]).copy()

        prophet_data['ds'] = prophet_data.apply(

            lambda row : datetime.datetime.strptime(str(row.name) + '-12-31', '%Y-%m-%d'), 

            axis=1

        )

        prophet_data.rename(columns={self.d:'y'}, inplace=True)

        train_prophet = prophet_data[prophet_data.apply(lambda row : int(row['ds'].year) <= 2000, axis=1)]

        test_prophet = prophet_data[prophet_data.apply(lambda row : int(row['ds'].year) > 2000, axis=1)]

        self.train = train_prophet.copy()

        self.test = test_prophet.copy()

        

        self.model = Prophet()

        self.model.fit(self.train)

        self.future = self.model.make_future_dataframe(periods=18, freq='A', include_history=True)

        self.forecast = self.model.predict(self.future)    

        

    def plot_forecast(self):

        return self.model.plot(fcst=self.forecast, ylabel='y ({})'.format(self.d))
models = {}

for d in directions:

    m = TemporaryClass(d)

    models[d] = m

    fig = m.plot_forecast()
r2 = {}

for d in directions:

    y_true = [filtered_frame[d].loc[j] for j in range(2001, 2018)]

    y_pred = list(models[d].forecast.tail(17)['yhat'])

    r2[d] = abs(r2_score(y_true=y_true, y_pred=y_pred))

r2
plt.figure(figsize=(10,6))

plt.grid()

plt.bar(range(len(r2)), list(r2.values()), align='center')

plt.xticks(range(len(r2)), list(r2.keys()))