import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import pandas as pd
from matplotlib import pyplot
df = pd.read_csv('../input/gpx-tracks-from-hikr.org.csv')
df.head(n=2)
df['avg_speed'] = df['length_3d']/df['moving_time']
df['difficulty_num'] = df['difficulty'].map(lambda x: int(x[1])).astype('int32')
df.describe()
# drop na values
df.dropna()
df = df[df['avg_speed'] < 2.5] # an avg of > 2.5m/s is probably not a hiking activity
def retain_values(df, column, min_quartile, max_quartile):
    q_min, q_max = df[column].quantile([min_quartile, max_quartile])
    print("Keeping values between {} and {} of column {}".format(q_min, q_max, column))
    return df[(df[column] > q_min) & (df[column] < q_max)]

# drop elevation outliers
df = retain_values(df, 'min_elevation', 0.01, 1)
df.corr()
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

y = df.reset_index()['moving_time']
x = df.reset_index()[['downhill', 'uphill', 'length_3d', 'max_elevation']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

lasso = Lasso()
lasso.fit(x_train, y_train)
print("Coefficients: {}".format(lasso.coef_))
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_pred_lasso = lasso.predict(x_test)
r2 = r2_score(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)
r2 = r2_score(y_test, y_pred_gbr)
mse = mean_squared_error(y_test, y_pred_gbr)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))
from keras import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()
model.add(Dense(12, input_shape=(4,)))
model.add(Dense(5, input_shape=(4,)))
model.add(Dense(1))
model.compile(optimizer=Adam(0.001), loss='mse')
hist = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.15, 
          callbacks=[
            ModelCheckpoint(filepath='./keras-model.h5', save_best_only=True),
            EarlyStopping(patience=2),
            ReduceLROnPlateau()
          ],
          verbose=1
)
model.load_weights(filepath='./keras-model.h5')
y_pred_keras = model.predict(x_test)

r2 = r2_score(y_test, y_pred_keras)
mse = mean_squared_error(y_test, y_pred_keras)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))
import numpy as np

combined = (y_pred_keras[:,0] + y_pred_gbr * 2) / 3.0
r2 = r2_score(y_test, combined)
mse = mean_squared_error(y_test, combined)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))
c = pd.DataFrame([combined, y_pred_keras[:,0], y_pred_lasso, y_pred_gbr, y_test]).transpose()
c.columns = ['combined', 'keras', 'lasso', 'tree', 'test']
c['diff_minutes'] = (c['test'] - c['combined']) / 60
c.describe()