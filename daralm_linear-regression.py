import pandas

import re

import numpy

from matplotlib import pyplot

from sklearn import linear_model

from sklearn import model_selection

from sklearn import preprocessing

from sklearn import metrics
df_raw = pandas.read_csv('/kaggle/input/hourly-traffic-volume-in-victoria/TYPICAL_HOURLY_VOLUME_DATA.csv')
df_melt = pandas.melt(df_raw,

                      id_vars=df_raw.columns[:7],

                      value_vars = df_raw.columns[7:],

                      var_name='hour_in_day',

                      value_name='traffic_count')
df_melt.head()
df_base = df_melt.loc[(df_melt['HMGNS_LNK_DESC'].notna()) &

                      (df_melt['HMGNS_LNK_DESC'].str.contains('^[ ]MONASH FREEWAY',flags=re.I, regex=True)), :].copy()

df_base.head()
df_base.shape
df_base.describe()
df_base['direction'] = df_base['FLOW'].apply(lambda x: 'Outbound' if x in ['EAST BOUND', 'SOUTH EAST BOUND', 'SOUTH BOUND']

                                            else 'Inbound')
df_pivot = df_base.pivot_table(columns='direction', index=['DOW', 'hour_in_day', 'HMGNS_LNK_DESC'], values='traffic_count', aggfunc='mean')

df_pivot.reset_index(inplace=True)

df_pivot['hour_in_day'] = df_pivot['hour_in_day'].str.split(':', expand=True).astype('int64')

df_pivot.head()
le = preprocessing.LabelEncoder()

le.fit(df_pivot['HMGNS_LNK_DESC'])

df_pivot['HMGNS_LNK_DESC_num'] = le.transform(df_pivot['HMGNS_LNK_DESC'])

x = df_pivot.drop(columns=['Outbound', 'Inbound', 'HMGNS_LNK_DESC'])

y = df_pivot['Outbound']

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

print(y_test.shape)

print(y_prediction.shape)

model_eval = numpy.sqrt(metrics.mean_squared_error(y_test, y_prediction))

print(model_eval)



import seaborn



lp = seaborn.lineplot(data=df_pivot.loc[df_pivot['HMGNS_LNK_DESC'] == ' MONASH FREEWAY btwn Toorak Rd & Bourke Rd', :],

                      x='hour_in_day', y='Inbound', hue='DOW', markers=True, dashes=df_pivot, estimator=None,

                      legend='full', palette=seaborn.color_palette("muted", n_colors=7))

pyplot.show()
df_pivot.corr().style.background_gradient(cmap='coolwarm')
df_pivot.head().style.background_gradient(cmap='coolwarm')