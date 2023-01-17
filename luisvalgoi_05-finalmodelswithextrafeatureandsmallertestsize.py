import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/2-years-restaurant-sale-with-multiple-external-var/sales_with_extra_features.csv', ',')
df = df.rename(columns={'IS_SEMANA_PAGAMENTO': 'SEMANA_PAGAMENTO'})
df = df.query('VENDAS < 200 and VENDAS > 70') # reading and removing outliers
df.info()
df.describe()
df.median()
print(df.isnull().values.any())
print(df.isna().values.any())
import matplotlib.style as style
import matplotlib.pyplot as plt

plt.rcParams['axes.facecolor'] = '#F1EFEF'

ax = df.plot(y=['VENDAS', 'TEMPERATURA'], x='DATA', figsize=(14, 5))
plt.legend(loc=1)
plt.title('AMOSTRAGEM DE TODOS OS DIAS ABERTOS ENTRE VENDAS E TEMPERATURA')
plt.xlabel('DATA')
plt.ylabel('VENDAS')
plt.show()    

df_quarter = pd.DataFrame({'DATA': pd.to_datetime(df['DATA']), 'TEMPERATURA': df['TEMPERATURA'], 'VENDAS': df['VENDAS']})

df_quarter.groupby(pd.to_datetime(df['DATA']).dt.to_period('Q'))['VENDAS'].agg('mean').plot(y='VENDAS', x='DATA', kind='bar', figsize=(14, 3))
plt.title('VENDAS TRIMESTRAIS')
plt.xlabel('VENDAS')
plt.ylabel('TRIMESTRES')
plt.show()

df_quarter.groupby(pd.to_datetime(df['DATA']).dt.to_period('Q'))['VENDAS'].agg('std').plot(y='VENDAS', x='DATA', kind='bar', figsize=(14, 3))
plt.title('DESVIO PADRAO DAS VENDAS TRIMESTRAIS')
plt.xlabel('MESES')
plt.ylabel('VENDAS')
plt.show()

ax = df_quarter.groupby(pd.to_datetime(df['DATA']).dt.to_period('M'))['VENDAS'].agg('mean').plot(y='VENDAS', x='DATA', kind='bar', figsize=(14, 3))
plt.title('VENDAS TRIMESTRAIS')
plt.xlabel('VENDAS')
plt.ylabel('TRIMESTRES')
plt.show()

mm = df_quarter.groupby(pd.to_datetime(df['DATA']).dt.to_period('M'))['VENDAS'].agg('std').plot(y='VENDAS', x='DATA', kind='bar', figsize=(14, 3))
plt.title('DESVIO PADRAO DAS VENDAS MENSAIS')
plt.xlabel('MESES')
plt.ylabel('VENDAS')
plt.show()
plt.rcParams['axes.facecolor'] = '#F1EFEF'
df.plot(y='VENDAS', x='DATA', figsize=(10, 5), kind='box')
plt.title('AMOSTRAGEM DA MEDIANA DE TODOS OS DIAS ABERTOS')
plt.show()
plt.rcParams['axes.facecolor'] = '#F1EFEF'
df.plot(y='VENDAS', x='DATA', figsize=(10, 5), kind='kde')
plt.legend(loc=1)
plt.title('ESTIMATIVA DE DENSIDADE DE KERNEL DE VENDAS')
plt.xlabel('VENDAS')
plt.ylabel('DENSIDADE')
plt.show()
plt.rcParams['axes.facecolor'] = '#F1EFEF'
query = (df['DATA'] >= "2019-06-01") & (df['DATA'] <= "2019-06-30")
df2 = df.loc[query]
df2.sort_values(by=['DATA'], inplace=True, ascending=True)
df2.plot(y=['VENDAS', 'PRECIPITACAO'], x='DATA', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.legend(loc=3)
plt.title('AMOSTRAGEM DO MÊS DE MAIO DE 2019')
plt.xlabel('DATA')
plt.ylabel('VENDAS')
plt.show()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# featured engineering
X = df.drop(columns=['DATA', 'VENDAS', 'SEMANA_DE_NAO_PAGAMENTO', 'SEMANA_PAGAMENTO', 'FERIADO'])
y = df.drop(columns=['DATA', 'FDS', 'DS', 'DATA_FESTIVA', 'VESPERA_DATA_FESTIVA', 'POS_DATA_FESTIVA', 'DATA_NAO_FESTIVA', 'FERIADO', 'NAO_FERIADO', 'SEMANA_PAGAMENTO', 'SEMANA_DE_NAO_PAGAMENTO', 'BAIXA_TEMPORADA', 'ALTA_TEMPORADA', 'QTD_CONCORRENTES', 'PRECIPITACAO', 'TEMPERATURA', 'UMIDADE', 'VENDAS_MEDIA_TRIM', 'VENDAS_MEDIA_MES', 'VENDAS_STD_TRIM', 'VENDAS_STD_MES'])

# shuffled and splitted into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
y_train = np.array(y_train)
y_test = np.array(y_test)

#feature scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#F1EFEF'

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=['FDS', 'DS', 'DATA_FESTIVA', 'VESPERA_DATA_FESTIVA', 'POS_DATA_FESTIVA', 'DATA_NAO_FESTIVA', 'NAO_FERIADO', 'BAIXA_TEMPORADA', 'ALTA_TEMPORADA', 'QTD_CONCORRENTES', 'PRECIPITACAO', 'TEMPERATURA', 'UMIDADE', 'VENDAS_MEDIA_TRIM', 'VENDAS_MEDIA_MES', 'VENDAS_STD_TRIM', 'VENDAS_STD_MES'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5))

sns.kdeplot(df['FDS'], bw=1.5, ax=ax1)
sns.kdeplot(df['DS'], bw=1.5, ax=ax1)
sns.kdeplot(df['DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['VESPERA_DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['POS_DATA_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['DATA_NAO_FESTIVA'], bw=1.5, ax=ax1)
sns.kdeplot(df['NAO_FERIADO'], bw=1.5, ax=ax1)
sns.kdeplot(df['BAIXA_TEMPORADA'], bw=1.5, ax=ax1)
sns.kdeplot(df['ALTA_TEMPORADA'], bw=1.5, ax=ax1)
sns.kdeplot(df['QTD_CONCORRENTES'], bw=1.5, ax=ax1)
sns.kdeplot(df['PRECIPITACAO'], bw=1.5, ax=ax1)
sns.kdeplot(df['TEMPERATURA'], bw=1.5, ax=ax1)
sns.kdeplot(df['UMIDADE'], bw=1.5, ax=ax1)
sns.kdeplot(df['VENDAS_MEDIA_TRIM'], bw=1.5, ax=ax1)
sns.kdeplot(df['VENDAS_MEDIA_MES'], bw=1.5, ax=ax1)
sns.kdeplot(df['VENDAS_STD_TRIM'], bw=1.5, ax=ax1)
sns.kdeplot(df['VENDAS_STD_MES'], bw=1.5, ax=ax1)

sns.kdeplot(scaled_df['FDS'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DS'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VESPERA_DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['POS_DATA_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['DATA_NAO_FESTIVA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['NAO_FERIADO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['BAIXA_TEMPORADA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['ALTA_TEMPORADA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['QTD_CONCORRENTES'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['PRECIPITACAO'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['TEMPERATURA'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['UMIDADE'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VENDAS_MEDIA_TRIM'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VENDAS_MEDIA_MES'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VENDAS_STD_TRIM'], bw=1.5, ax=ax2)
sns.kdeplot(scaled_df['VENDAS_STD_MES'], bw=1.5, ax=ax2)

plt.show()
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
# imports
from sklearn.ensemble import GradientBoostingRegressor
plt.rcParams['axes.facecolor'] = '#F1EFEF'

# model
gb = GradientBoostingRegressor(alpha=0.1, ccp_alpha=0.1, criterion='mse',
                          init=None, learning_rate=0.1, loss='ls',
                          max_depth=3, max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=12,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

# cross validation
scores = cross_val_score(gb, X, y, scoring='neg_root_mean_squared_error', cv=5)
scores_mae = cross_val_score(gb, X, y, scoring='neg_mean_absolute_error', cv=5)
predicted = cross_val_predict(gb, X, y, cv=5)
print(f'RMSE (cross validation): %.2f' % np.abs(scores).min())
print(f'MAE (cross validation): %.2f' % np.abs(scores_mae).min())

# manual training
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(f'RMSE: %.2f' % metrics.mean_squared_error(y_test, y_pred, squared=False))
print(f'MAE: %.2f' % metrics.mean_absolute_error(y_test, y_pred))

# plot
df_comparison = pd.DataFrame({'ATUAL': pd.DataFrame(y_test).values.flatten(), 'PREVISTO': y_pred.flatten()})
df_comparison.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

# permutation
perm = PermutationImportance(gb, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=X.columns.values.tolist())
# imports 
from sklearn.neural_network import MLPRegressor
plt.rcParams['axes.facecolor'] = '#F1EFEF'

# model
nn = MLPRegressor(activation='identity', alpha=0.002, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(244,), learning_rate='constant',
             learning_rate_init=0.001, max_fun=15000, max_iter=800,
             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
             power_t=0.5, random_state=3, shuffle=True, solver='adam',
             tol=0.0001, validation_fraction=0.1, verbose=False,
             warm_start=False)

# cross validation
scores = cross_val_score(nn, X, y, scoring='neg_root_mean_squared_error', cv=5)
scores_mae = cross_val_score(nn, X, y, scoring='neg_mean_absolute_error', cv=5)
predicted = cross_val_predict(nn, X, y, cv=5)
print(f'RMSE (cross validation): %.2f' % np.abs(scores).min())
print(f'MAE (cross validation): %.2f' % np.abs(scores_mae).min())

# manual training
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
print(f'RMSE: %.2f' % metrics.mean_squared_error(y_test, y_pred, squared=False))
print(f'MAE: %.2f' % metrics.mean_absolute_error(y_test, y_pred))

# plot
df1 = pd.DataFrame({'ATUAL': pd.DataFrame(y_test).values.flatten(), 'PREVISTO': y_pred.flatten()})
df1.plot(kind='line', figsize=(10, 5), linestyle='solid', linewidth=1, markersize=2, style="-o")
plt.show()

# permutation
perm = PermutationImportance(nn, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=X.columns.values.tolist())
# imports
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import cross_val_score # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

## Deep-learing imports
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

plt.rcParams['axes.facecolor'] = '#F1EFEF'

# reshape input to be 3D as expected by LSTMs [samples, timesteps, features].
train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
train_y = y_train.reshape((pd.DataFrame(y_train).shape[0], -1))
test_y = y_test.reshape((pd.DataFrame(y_test).shape[0], -1))

# create model
model = Sequential()
model.add(LSTM(484, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(train_X, train_y, epochs=100, batch_size=70, shuffle=False)

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], train_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -16:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -16:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = mean_squared_error(inv_y, inv_yhat, squared=False)
mae = mean_absolute_error(inv_y, inv_yhat)
print('RMSE: %.2f' % rmse)
print('MAE: %.2f' % mae)

# plot
aa=[x for x in range(inv_yhat.size)]
plt.plot(aa, inv_y, marker='.', label="ATUAL")
plt.plot(aa, inv_yhat, 'r', color='green', label="PREVISTO")
plt.ylabel('VENDAS', size=15)
plt.xlabel('DATA', size=15)
plt.legend(fontsize=15)
plt.show()