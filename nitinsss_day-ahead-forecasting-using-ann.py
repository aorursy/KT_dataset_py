import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
df = pd.read_csv(r'/kaggle/input/russian-wholesale-electricity-market/RU_Electricity_Market_PZ_dayahead_price_volume.csv')
df.head()
df.info()
df['price_eur'] = df['price_eur'].apply(lambda x : float(x.replace(',', '')))



df['price_sib'] = df['price_sib'].apply(lambda x : float(x.replace(',', '')))
df['datetime'] = pd.to_datetime(df['timestep'])



df.drop(['timestep'], axis=1, inplace=True)



df.set_index(['datetime'], drop=True, inplace=True)
df.info()
plt.figure(figsize = (9,9))



plt.subplot(4, 1, 1)

df['consumption_eur'].plot()



plt.subplot(4, 1, 2)

df['consumption_sib'].plot()



plt.subplot(4, 1, 3)

df['price_eur'].plot()



plt.subplot(4, 1, 4)

df['price_sib'].plot()
from statsmodels.tsa import stattools



acf_cons_eur = stattools.acf(df['consumption_eur'], unbiased=True, nlags=100)

acf_cons_sib = stattools.acf(df['consumption_sib'], unbiased=True, nlags=100)

acf_price_eur = stattools.acf(df['price_eur'], unbiased=True, nlags=100)

acf_price_sib = stattools.acf(df['price_sib'], unbiased=True, nlags=100)
plt.figure(figsize = (9,11))



plt.subplot(4, 1, 1)

pd.Series(acf_cons_eur).plot()

plt.grid()



plt.subplot(4, 1, 2)

pd.Series(acf_cons_sib).plot()

plt.grid()



plt.subplot(4, 1, 3)

pd.Series(acf_price_eur).plot()

plt.grid()



plt.subplot(4, 1, 4)

pd.Series(acf_price_sib).plot()

plt.grid()
autocorrs = pd.Series(acf_cons_eur)



list_of_regressors_cons_eur = autocorrs.loc[autocorrs > 0.9].index



list_of_regressors_cons_eur = list_of_regressors_cons_eur[1:11]



list_of_regressors_cons_eur
autocorrs = pd.Series(acf_cons_sib)



list_of_regressors_cons_sib = autocorrs.loc[autocorrs > 0.9].index



list_of_regressors_cons_sib = list_of_regressors_cons_sib[1:11]



list_of_regressors_cons_sib
autocorrs = pd.Series(acf_price_eur)



list_of_regressors_price_eur = autocorrs.loc[autocorrs > 0.8].index



list_of_regressors_price_eur = list_of_regressors_price_eur[1:11]



list_of_regressors_price_eur
autocorrs = pd.Series(acf_price_sib)



list_of_regressors_price_sib = autocorrs.loc[autocorrs > 0.85].index



list_of_regressors_price_sib = list_of_regressors_price_sib[1:11]



list_of_regressors_price_sib
def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :

    

    """

    Ensure that the index is of datetime type

    Creates features with previous time instant values

    """

        

    list_of_prev_t_instants.sort_values

    start = list_of_prev_t_instants[-1] 

    end = len(df)

    df['datetime'] = df.index

    df.reset_index(drop=True)



    df_copy = df[start:end]

    df_copy.reset_index(inplace=True, drop=True)



    for attribute in attribute :

            foobar = pd.DataFrame()



            for prev_t in list_of_prev_t_instants :

                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])

                new_col.reset_index(drop=True, inplace=True)

                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)

                foobar = pd.concat([foobar, new_col], sort=False, axis=1)



            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)

            

    df_copy.set_index(['datetime'], drop=True, inplace=True)

    return df_copy
cons_eur = df.loc[:, [ 'consumption_eur']]

df_consum_eur = create_regressor_attributes(cons_eur, ['consumption_eur'], list_of_regressors_cons_eur)

df_consum_eur.head(3)
cons_sib = df.loc[:, [ 'consumption_sib']]

df_consum_sib = create_regressor_attributes(cons_sib, ['consumption_sib'], list_of_regressors_cons_sib)

df_consum_sib.head(3)
df_after_2010 = df.loc[df.index.year >= 2010]

df_after_2010.head(3)
price_eur = df_after_2010.loc[:, [ 'price_eur']]

df_price_eur = create_regressor_attributes(price_eur, ['price_eur'], list_of_regressors_price_eur)

df_price_eur.head(3)
price_sib = df_after_2010.loc[:, [ 'price_sib']]

df_price_sib = create_regressor_attributes(price_sib, ['price_sib'], list_of_regressors_price_sib)

df_price_sib.head(3)
def train_test_valid_split_plus_scaling(df, valid_set_size, test_set_size):

    

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import MinMaxScaler

    

    df_copy = df.reset_index(drop=True)

    

    df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]

    df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]



    df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]

    df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]





    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]

    X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]

    X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

    

    global Target_scaler

    

    Target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

    Feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

    

    X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))

    X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))

    X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))

    

    y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1,1))

    y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1,1))

    y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1,1))

    

    print('Shape of training inputs, training target:', X_train_scaled.shape, y_train_scaled.shape)

    print('Shape of validation inputs, validation target:', X_valid_scaled.shape, y_valid_scaled.shape)

    print('Shape of test inputs, test targets:', X_test_scaled.shape, y_test_scaled.shape)



    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train_scaled, y_valid_scaled, y_test_scaled
valid_set_size = 0.1

test_set_size = 0.1

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(df_consum_eur, 

                                                                                         valid_set_size, 

                                                                                        test_set_size)
from tensorflow.keras.layers import Input, Dense, Dropout

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Model
input_layer = Input(shape=9, dtype='float32')

dense1 = Dense(27, activation='linear')(input_layer)

dense2 = Dense(18, activation='linear')(dense1)

dense3 = Dense(18, activation='linear')(dense2)

dropout_layer = Dropout(0.2)(dense2)

output_layer = Dense(1, activation='linear')(dropout_layer)



consum_eur_model = Model(inputs=input_layer, outputs=output_layer)

consum_eur_model.compile(loss='mean_squared_error', optimizer='adam')

consum_eur_model.summary()
consum_eur_model.fit(x=X_train, y=y_train, batch_size=100, epochs=25, verbose=1, validation_data=(X_valid, y_valid), shuffle=True)
def run_model_on_test_set(model, df, X_test, y_test, Target_scaler):

    

    #Recall that we have already defined "Target_scaler" as global in the 'split n scale' function earlier 

    

    y_pred = model.predict(X_test)

    y_pred_rescaled = Target_scaler.inverse_transform(y_pred)

    

    y_test_rescaled =  Target_scaler.inverse_transform(y_test)

    

    from sklearn.metrics import r2_score

    score = r2_score(y_test_rescaled, y_pred_rescaled)

    print('R-squared score for the test set:', round(score,4))

    

    y_actual = pd.DataFrame(y_test_rescaled, columns=['Actual'])

    y_actual.set_index(df.index[ int(np.floor((1- test_set_size)*len(df))) : ], inplace=True)



    y_hat = pd.DataFrame(y_pred_rescaled, columns=['Predicted'])

    y_hat.set_index(df.index[ int(np.floor((1- test_set_size)*len(df))) : ], inplace=True)

    

    plt.figure(figsize=(7, 5))

    plt.plot(y_actual[500:600], linestyle='solid', color='r')   #plotting only a few values for better visibility

    plt.plot(y_hat[500:600], linestyle='dashed', color='b')

    plt.legend(['Actual','Predicted'], loc='best', prop={'size': 14})

    plt.title('{}'.format(df.columns[0]), weight='bold', fontsize=16)

    plt.ylabel('Value', weight='bold', fontsize=14)

    plt.xlabel('Date', weight='bold', fontsize=14)

    plt.xticks(weight='bold', fontsize=12, rotation=45)

    plt.yticks(weight='bold', fontsize=12)

    plt.grid(color = 'y', linewidth='0.5')

    plt.show()
run_model_on_test_set(consum_eur_model, df_consum_eur, X_test, y_test, Target_scaler)
valid_set_size = 0.1

test_set_size = 0.1

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(df_consum_sib, 

                                                                                         valid_set_size, 

                                                                                        test_set_size)



input_layer = Input(shape=10, dtype='float32')        #Refer again the number of regressors that we had fixed for this particular attribute

dense1 = Dense(30, activation='linear')(input_layer)

dense2 = Dense(20, activation='linear')(dense1)

dense3 = Dense(20, activation='linear')(dense2)

dropout_layer = Dropout(0.2)(dense2)

output_layer = Dense(1, activation='linear')(dropout_layer)



consum_sib_model = Model(inputs=input_layer, outputs=output_layer)

consum_sib_model.compile(loss='mean_squared_error', optimizer='adam')

consum_sib_model.summary()



consum_sib_model.fit(x=X_train, y=y_train, batch_size=100, epochs=25, verbose=1, validation_data=(X_valid, y_valid), shuffle=True)



run_model_on_test_set(consum_sib_model, df_consum_sib, X_test, y_test, Target_scaler)
valid_set_size = 0.1

test_set_size = 0.1

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(df_price_eur, 

                                                                                         valid_set_size, 

                                                                                        test_set_size)



input_layer = Input(shape=10, dtype='float32')        

dense1 = Dense(30, activation='linear')(input_layer)

dense2 = Dense(20, activation='linear')(dense1)

dense3 = Dense(20, activation='linear')(dense2)

dropout_layer = Dropout(0.2)(dense2)

output_layer = Dense(1, activation='linear')(dropout_layer)



price_eur_model = Model(inputs=input_layer, outputs=output_layer)

price_eur_model.compile(loss='mean_squared_error', optimizer='adam')

price_eur_model.summary()



price_eur_model.fit(x=X_train, y=y_train, batch_size=50, epochs=25, verbose=1, validation_data=(X_valid, y_valid), shuffle=True)



run_model_on_test_set(price_eur_model, df_price_eur, X_test, y_test, Target_scaler)
valid_set_size = 0.1

test_set_size = 0.1

X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_plus_scaling(df_price_sib, 

                                                                                         valid_set_size, 

                                                                                        test_set_size)



input_layer = Input(shape=10, dtype='float32')        

dense1 = Dense(30, activation='linear')(input_layer)

dense2 = Dense(20, activation='linear')(dense1)

dense3 = Dense(20, activation='linear')(dense2)

dropout_layer = Dropout(0.2)(dense2)

output_layer = Dense(1, activation='linear')(dropout_layer)



price_sib_model = Model(inputs=input_layer, outputs=output_layer)

price_sib_model.compile(loss='mean_squared_error', optimizer='adam')

price_sib_model.summary()



price_sib_model.fit(x=X_train, y=y_train, batch_size=50, epochs=25, verbose=1, validation_data=(X_valid, y_valid), shuffle=True)



run_model_on_test_set(price_sib_model, df_price_sib, X_test, y_test, Target_scaler)