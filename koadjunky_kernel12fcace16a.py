import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df = pd.concat([train, test], axis=0, sort=True)
df.head()
df.info()
df.count()
df1 = df.drop(['dteday', 'casual', 'registered', 'year'], axis=1)
df1.head()
# Want this re-entrant

df1['season'] = df['season'] - 1
df1['mnth'] = df['mnth'] - 1
df1['weathersit'] = df['weathersit'] - 1
df1.head()
continuous = ['temp', 'atemp', 'hum', 'windspeed']
scaler = StandardScaler()
for var in continuous:
  df1[var] = df[var].astype('float64')
  df1[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

df1.head()
train = df1[df1['cnt'].notna()]
train
test = df1[df1['cnt'].isna()]
test
x_train = train.drop(['cnt'], axis=1)
y_train = train['cnt']
x_test = test.drop(['cnt'], axis=1)
y_test = test['cnt']
season = tf.feature_column.categorical_column_with_identity("season", 4)
month = tf.feature_column.categorical_column_with_identity("mnth", 12)
hour = tf.feature_column.categorical_column_with_identity("hr", 24)
holiday = tf.feature_column.categorical_column_with_identity("holiday", 2)
weekday = tf.feature_column.categorical_column_with_identity("weekday", 7)
workingday = tf.feature_column.categorical_column_with_identity("workingday", 2)
weathersit = tf.feature_column.categorical_column_with_identity("weathersit", 4)
temp = tf.feature_column.numeric_column("temp")
atemp = tf.feature_column.numeric_column("atemp")
hum = tf.feature_column.numeric_column("hum")
windspeed = tf.feature_column.numeric_column("windspeed")

feat_cols = [season, month, hour, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

#model = tf.estimator.LinearRegressor(feature_columns=feat_cols) # 146
model = tf.estimator.BoostedTreesRegressor(feature_columns=feat_cols, n_batches_per_layer=30) # 74
feat_cols2 = [
              tf.feature_column.indicator_column(season),
              tf.feature_column.indicator_column(month),
              tf.feature_column.indicator_column(hour),
              tf.feature_column.indicator_column(holiday),
              tf.feature_column.indicator_column(weekday),
              tf.feature_column.indicator_column(workingday),
              tf.feature_column.indicator_column(weathersit),
              temp,
              atemp,
              hum,
              windspeed,
]
#model = tf.estimator.DNNRegressor(feature_columns=feat_cols2, hidden_units=[256, 256, 256, 256, 256]) # 67.2
#model = tf.estimator.DNNRegressor(feature_columns=feat_cols2, hidden_units=[512, 512, 512, 512, 512]) # 66
#model = tf.estimator.DNNRegressor(feature_columns=feat_cols2, hidden_units=[512, 512, 512, 512, 512], optimizer='Adam') # 66
#model = tf.estimator.DNNRegressor(feature_columns=feat_cols2, hidden_units=[512, 512], optimizer='Adam') # 71

model.train(input_fn=input_func, max_steps=20000)
pred_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=len(x_test), shuffle=False)

predictions = list(model.predict(input_fn=pred_func))
final_preds = []
for pred in predictions:
  final_preds.append(pred['predictions'][0])
final_preds1 = [int(x) if x > 0 else 0 for x in final_preds]


final_preds1
submission = x_test[['instant']]
submission['cnt'] = final_preds1
submission
submission.to_csv('answer_btr.csv', index=False)