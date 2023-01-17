import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
raw_data = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')

raw_data.head()
raw_data.info()
raw_data.isna().sum()
cleaned_data = raw_data.copy()

cleaned_data['jersey'] = cleaned_data['jersey'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['team'] = cleaned_data['team'].fillna('no team')   # fill all n/a with 'no team' string

cleaned_data['height'] = cleaned_data['height'].apply(lambda x: float(x[2+x.find('/'):])) # convert to meters

cleaned_data['weight'] = cleaned_data['weight'].apply(lambda x: float(x[2+x.find('/'):-4])) # convert to kg

cleaned_data['salary'] = cleaned_data['salary'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['draft_round'] = cleaned_data['draft_round'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['draft_peak'] = cleaned_data['draft_peak'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['college'] = cleaned_data['college'].fillna('no college')

cleaned_data['experience_years'] = 2020 - cleaned_data['draft_year']

cleaned_data = cleaned_data.drop(['draft_year'], axis=1)



# change bday on age

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: x[-2:])

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: int('20'+x) if x[0] == '0' else int('19'+x))

cleaned_data['age'] = 2020 - cleaned_data['b_day']

cleaned_data = cleaned_data.drop(['b_day'], axis=1)
cleaned_data
labelencoder = LabelEncoder()

cleaned_data['position_cat'] = labelencoder.fit_transform(cleaned_data['position'])
labelencoder = LabelEncoder()

cleaned_data['team_cat'] = labelencoder.fit_transform(cleaned_data['team'])
labelencoder = LabelEncoder()

cleaned_data['country_cat'] = labelencoder.fit_transform(cleaned_data['country'])
labelencoder = LabelEncoder()

cleaned_data['college_cat'] = labelencoder.fit_transform(cleaned_data['college'])
cleaned_data
plt.figure(figsize=(15,8))

sns.heatmap(cleaned_data.corr(), annot=True, linewidths=0.5, linecolor='black', cmap='coolwarm')

plt.show()
features = ['rating', 'draft_peak', 'experience_years', 'age', 'team_cat', 'country_cat', 'position_cat', 'draft_round']

label = 'salary'



x, y = cleaned_data[features], cleaned_data['salary']



normalizer = Normalizer().fit(x)

x = normalizer.transform(x)

x = np.array(x)

y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
reg = LinearRegression()

reg.fit(x_train, y_train)



print("error: ", np.sqrt(mean_squared_error(y_test, reg.predict(x_test))))
model = XGBRegressor( 

    n_estimators = 300,

    learning_rate=0.04,

    colsample_bytree=0.9, 

    min_child_weight=3,

    objective='reg:squarederror',

    max_depth = 2,

    subsample = 0.63,

    eta = 0.1,

    seed=0)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    early_stopping_rounds=10,

    eval_set=[(x_test,y_test)],

    verbose=False)
print("error: ", np.sqrt(mean_squared_error(y_test, model.predict(x_test))))

def build_and_compile_model():

    model = keras.Sequential([

      layers.Dense(32, activation='relu'),

      layers.Dense(1)])



    model.compile(loss='mean_absolute_error',

                optimizer=tf.keras.optimizers.Adam(0.001))

    return model
dnn_model = build_and_compile_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = dnn_model.fit(

    x_train, y_train,

    validation_split=0.2,

    verbose=0, epochs=200, callbacks=[early_stop])
print("error: ", np.sqrt(mean_squared_error(y_test, dnn_model.predict(x_test).flatten())))