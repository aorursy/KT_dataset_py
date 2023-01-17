import math

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import pandas as pd

from google.cloud import bigquery

from bq_helper import BigQueryHelper

from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, Dense, LSTM, BatchNormalization

from keras.preprocessing import sequence

# from keras.layers import Input, Dense, Activation

# from keras.callbacks import ModelCheckpoint

# from keras.optimizers import Adam, SGD

# from keras.utils import to_categorical

from keras import metrics

from keras import models

from keras import layers

from keras import optimizers

    

INDICATORS = """

    SELECT

       indicator_name

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND (indicator_value > 0 OR indicator_value < 0)

      AND indicator_name NOT LIKE 'GDP%'

      AND year = 1960

      AND indicator_name NOT LIKE 'International migrant stock, total'

      AND indicator_name NOT LIKE 'Net official development assistance received (current US$)'

      AND indicator_name NOT LIKE 'Terms of trade adjustment (constant LCU)'

      AND indicator_name NOT LIKE 'Total reserves minus gold (current US$)'

      AND indicator_name NOT LIKE 'Trademark applications, direct resident'

      AND indicator_name NOT LIKE 'Total reserves minus gold (current US$)'

      AND indicator_name NOT LIKE 'Trademark applications, direct resident'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in East Asia & Pacific (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Denmark (current US$)'

      AND indicator_name NOT LIKE 'CO2 emissions from solid fuel consumption (kt)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (constant LCU)'

      AND indicator_name NOT LIKE 'Net ODA received per capita (current US$)'

      AND indicator_name NOT LIKE 'Gross capital formation (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Italy (current US$)'

      AND indicator_name NOT LIKE 'CO2 emissions from liquid fuel consumption (kt)'

      AND indicator_name NOT LIKE 'Merchandise exports by the reporting economy, residual (% of total merchandise exports)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'GNI (constant LCU)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Patent applications, nonresidents'

      AND indicator_name NOT LIKE 'Total reserves (includes gold, current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Merchandise imports by the reporting economy, residual (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Net ODA received (% of gross capital formation)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in South Asia (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Exports of goods and services (current LCU)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (constant LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'Changes in inventories (current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (% of GDP)'

      AND indicator_name NOT LIKE 'Merchandise imports from high-income economies (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'CO2 emissions (kg per 2010 US$ of GDP)'

      AND indicator_name NOT LIKE 'CO2 emissions (kt)'

      AND indicator_name NOT LIKE 'External balance on goods and services (constant LCU)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies outside region (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Sweden (current US$)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (current LCU)'

      AND indicator_name NOT LIKE 'Changes in inventories (constant LCU)'

      AND indicator_name NOT LIKE 'CO2 emissions from solid fuel consumption (% of total)'

      AND indicator_name NOT LIKE 'External balance on goods and services (current LCU)'

      AND indicator_name NOT LIKE 'CO2 emissions from liquid fuel consumption (% of total)'

      AND indicator_name NOT LIKE 'Final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Age dependency ratio, old (% of working-age population)'

      AND indicator_name NOT LIKE 'External balance on goods and services (current US$)'

      AND indicator_name NOT LIKE 'Gross national expenditure (% of GDP)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (% of GDP)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (current LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'External balance on goods and services (% of GDP)'

      AND indicator_name NOT LIKE 'Gross capital formation (current LCU)'

      AND indicator_name NOT LIKE 'Final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'Gross national expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'Net official development assistance received (constant 2013 US$)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Gross capital formation (current US$)'

      AND indicator_name NOT LIKE 'Mortality rate, infant (per 1,000 live births)'

      AND indicator_name NOT LIKE 'Broad money to total reserves ratio'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (current LCU)'

      AND indicator_name NOT LIKE 'Merchandise exports (current US$)'

      AND indicator_name NOT LIKE 'Merchandise imports by the reporting economy (current US$)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, United States (current US$)'

      AND indicator_name NOT LIKE 'Exports of goods and services (constant LCU)'

      AND indicator_name NOT LIKE 'GNI per capita (current LCU)'

      AND indicator_name NOT LIKE 'Gross capital formation (constant LCU)'

      AND indicator_name NOT LIKE 'Gross national expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Exports as a capacity to import (constant LCU)'

      AND indicator_name NOT LIKE 'Net taxes on products (current US$)'

      AND indicator_name NOT LIKE 'Manufacturing, value added (constant LCU)'

      AND indicator_name NOT LIKE 'Wholesale price index (2010 = 100)'

      AND indicator_name NOT LIKE 'Gross fixed capital formation (constant LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Exports of goods and services (current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (constant LCU)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, United Kingdom (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Norway (current US$)'

      AND indicator_name NOT LIKE 'Net official development assistance and official aid received (current US$)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'GNI (current LCU)'

      AND indicator_name NOT LIKE 'Net ODA received (% of GNI)'

      AND indicator_name NOT LIKE 'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)'

      AND indicator_name NOT LIKE 'Population in largest city'

      AND indicator_name NOT LIKE 'Discrepancy in expenditure estimate of GDP (current LCU)'

      AND indicator_name NOT LIKE 'Gross fixed capital formation (current US$)'

      AND indicator_name NOT LIKE 'Changes in inventories (current LCU)'

      AND indicator_name NOT LIKE 'Life expectancy at birth, female (years)'

      AND indicator_name NOT LIKE 'Life expectancy at birth, male (years)'

      AND indicator_name NOT LIKE 'GNI (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Technical cooperation grants (BoP, current US$)'

      AND indicator_name NOT LIKE 'Grants, excluding technical cooperation (BoP, current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Total (current US$)'

      

    ORDER BY year

        """



QUERY = """

    SELECT

       year, indicator_name, indicator_value

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND (indicator_value > 0 OR indicator_value < 0)

      AND indicator_name NOT LIKE 'GDP%'

      AND year < 2014

    ORDER BY year

        """



TRAINTARGET = """

    SELECT

       indicator_value

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND indicator_name LIKE 'GDP per capita (current US$)'

      AND year < 2014

    ORDER BY year

        """



bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")



pd.options.display.max_rows=1000



client = bigquery.Client()

df = client.query(QUERY).to_dataframe()

df1 = client.query(INDICATORS).to_dataframe()

df_Obtained_Targets = client.query(TRAINTARGET).to_dataframe()



df2 = pd.merge(df, df1)

obtained_data = df2.values

obtained_targets = df_Obtained_Targets.values



arr = np.zeros((54, 90))

indicators = np.array([])



for i in obtained_data:

    n_row = i[0] -1960

    b_insert = True

    for j in range(len(indicators)):

        if i[1] == indicators[j]:

            arr[n_row, j] = i[2]

            b_insert = False

    if (b_insert):

        indicators = np.append(indicators, i[1])

        arr[n_row, len(indicators)-1] = i[2]

#print('indicators shape', indicators.shape)

for i in range(len(indicators)):

    if(indicators[i] == 'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)'):

        arr[1][i] = round((arr[0][i] + arr[2][i])/2, 15)

        

    if(indicators[i] == 'Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)'):

        arr[4][i] = round((arr[3][i] + arr[5][i])/2, 15)

        

    if(indicators[i] == 'Merchandise exports to low- and middle-income economies outside region (% of total merchandise exports)'):

        arr[1][i] = round((arr[0][i] + arr[2][i])/2, 15)



        

train_data = np.empty((0, 90))

train_targets = np.array([])

test_data = np.empty((0, 90))

test_targets = np.array([])





for i in range(len(arr)):

    if i < 49:

        train_data = np.concatenate((train_data,np.expand_dims(arr[i], axis = 0)))

        train_targets = np.append(train_targets, obtained_targets[i], axis = 0)

    else:

        test_data = np.concatenate((test_data,np.expand_dims(arr[i], axis = 0)))

        test_targets = np.append(test_targets, obtained_targets[i], axis = 0)



        



        

        

             

        

mean = train_data.mean(axis = 0)

train_data -= mean

std = train_data.std(axis=0)

train_data /= std



test_data -= mean

test_data /= std





train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))

test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))



    # LSTM MODEL

model = Sequential()

model.add(LSTM(2048))

model.add(Dense(2048, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics = ['mae'])

history = model.fit(train_data, train_targets, epochs=250, batch_size=90, validation_data=(test_data, test_targets))





mae = history.history['mean_absolute_error']

val_mae = history.history['val_mean_absolute_error']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, mae, 'b', label = 'Training mae')

plt.plot(epochs, val_mae, 'g', label = 'Validation mae')

plt.title('Training and validation mae')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'g', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()



#lstm 2048, 2048, 512 - epoch:120~130 min:2441 mean:2450

#lstm 2048, 1024, 256, 64 - epoch 32 min:1505 mean:1650

#lstm 2048, 512, 64 - epoch:200 min 3420 mean:3450
import math

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import pandas as pd

from google.cloud import bigquery

from bq_helper import BigQueryHelper

from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, Dense, LSTM, BatchNormalization

from keras.preprocessing import sequence

# from keras.layers import Input, Dense, Activation

# from keras.callbacks import ModelCheckpoint

# from keras.optimizers import Adam, SGD

# from keras.utils import to_categorical

from keras import metrics

from keras import models

from keras import layers

from keras import optimizers

    

INDICATORS = """

    SELECT

       indicator_name

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND (indicator_value > 0 OR indicator_value < 0)

      AND indicator_name NOT LIKE 'GDP%'

      AND year = 1960

      AND indicator_name NOT LIKE 'International migrant stock, total'

      AND indicator_name NOT LIKE 'Net official development assistance received (current US$)'

      AND indicator_name NOT LIKE 'Terms of trade adjustment (constant LCU)'

      AND indicator_name NOT LIKE 'Total reserves minus gold (current US$)'

      AND indicator_name NOT LIKE 'Trademark applications, direct resident'

      AND indicator_name NOT LIKE 'Total reserves minus gold (current US$)'

      AND indicator_name NOT LIKE 'Trademark applications, direct resident'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in East Asia & Pacific (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Denmark (current US$)'

      AND indicator_name NOT LIKE 'CO2 emissions from solid fuel consumption (kt)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (constant LCU)'

      AND indicator_name NOT LIKE 'Net ODA received per capita (current US$)'

      AND indicator_name NOT LIKE 'Gross capital formation (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Italy (current US$)'

      AND indicator_name NOT LIKE 'CO2 emissions from liquid fuel consumption (kt)'

      AND indicator_name NOT LIKE 'Merchandise exports by the reporting economy, residual (% of total merchandise exports)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'GNI (constant LCU)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Patent applications, nonresidents'

      AND indicator_name NOT LIKE 'Total reserves (includes gold, current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Merchandise imports by the reporting economy, residual (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Net ODA received (% of gross capital formation)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies in South Asia (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Exports of goods and services (current LCU)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (constant LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'Changes in inventories (current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (% of GDP)'

      AND indicator_name NOT LIKE 'Merchandise imports from high-income economies (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'CO2 emissions (kg per 2010 US$ of GDP)'

      AND indicator_name NOT LIKE 'CO2 emissions (kt)'

      AND indicator_name NOT LIKE 'External balance on goods and services (constant LCU)'

      AND indicator_name NOT LIKE 'Merchandise imports from low- and middle-income economies outside region (% of total merchandise imports)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Sweden (current US$)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (current LCU)'

      AND indicator_name NOT LIKE 'Changes in inventories (constant LCU)'

      AND indicator_name NOT LIKE 'CO2 emissions from solid fuel consumption (% of total)'

      AND indicator_name NOT LIKE 'External balance on goods and services (current LCU)'

      AND indicator_name NOT LIKE 'CO2 emissions from liquid fuel consumption (% of total)'

      AND indicator_name NOT LIKE 'Final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Age dependency ratio, old (% of working-age population)'

      AND indicator_name NOT LIKE 'External balance on goods and services (current US$)'

      AND indicator_name NOT LIKE 'Gross national expenditure (% of GDP)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (% of GDP)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (current LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'External balance on goods and services (% of GDP)'

      AND indicator_name NOT LIKE 'Gross capital formation (current LCU)'

      AND indicator_name NOT LIKE 'Final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'Gross national expenditure (constant LCU)'

      AND indicator_name NOT LIKE 'Net official development assistance received (constant 2013 US$)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Gross capital formation (current US$)'

      AND indicator_name NOT LIKE 'Mortality rate, infant (per 1,000 live births)'

      AND indicator_name NOT LIKE 'Broad money to total reserves ratio'

      AND indicator_name NOT LIKE 'Household final consumption expenditure (current LCU)'

      AND indicator_name NOT LIKE 'Merchandise exports (current US$)'

      AND indicator_name NOT LIKE 'Merchandise imports by the reporting economy (current US$)'

      AND indicator_name NOT LIKE 'Gross value added at factor cost (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, United States (current US$)'

      AND indicator_name NOT LIKE 'Exports of goods and services (constant LCU)'

      AND indicator_name NOT LIKE 'GNI per capita (current LCU)'

      AND indicator_name NOT LIKE 'Gross capital formation (constant LCU)'

      AND indicator_name NOT LIKE 'Gross national expenditure (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Exports as a capacity to import (constant LCU)'

      AND indicator_name NOT LIKE 'Net taxes on products (current US$)'

      AND indicator_name NOT LIKE 'Manufacturing, value added (constant LCU)'

      AND indicator_name NOT LIKE 'Wholesale price index (2010 = 100)'

      AND indicator_name NOT LIKE 'Gross fixed capital formation (constant LCU)'

      AND indicator_name NOT LIKE 'Household final consumption expenditure, etc. (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Exports of goods and services (current US$)'

      AND indicator_name NOT LIKE 'Final consumption expenditure, etc. (constant LCU)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, United Kingdom (current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Norway (current US$)'

      AND indicator_name NOT LIKE 'Net official development assistance and official aid received (current US$)'

      AND indicator_name NOT LIKE 'General government final consumption expenditure (current US$)'

      AND indicator_name NOT LIKE 'GNI (current LCU)'

      AND indicator_name NOT LIKE 'Net ODA received (% of GNI)'

      AND indicator_name NOT LIKE 'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)'

      AND indicator_name NOT LIKE 'Population in largest city'

      AND indicator_name NOT LIKE 'Discrepancy in expenditure estimate of GDP (current LCU)'

      AND indicator_name NOT LIKE 'Gross fixed capital formation (current US$)'

      AND indicator_name NOT LIKE 'Changes in inventories (current LCU)'

      AND indicator_name NOT LIKE 'Life expectancy at birth, female (years)'

      AND indicator_name NOT LIKE 'Life expectancy at birth, male (years)'

      AND indicator_name NOT LIKE 'GNI (constant 2010 US$)'

      AND indicator_name NOT LIKE 'Technical cooperation grants (BoP, current US$)'

      AND indicator_name NOT LIKE 'Grants, excluding technical cooperation (BoP, current US$)'

      AND indicator_name NOT LIKE 'Net bilateral aid flows from DAC donors, Total (current US$)'

    ORDER BY year

        """



QUERY = """

    SELECT

       year, indicator_name, indicator_value

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND (indicator_value > 0 OR indicator_value < 0)

      AND indicator_name NOT LIKE 'GDP%'

      AND year < 2014

    ORDER BY year

        """



TRAINTARGET = """

    SELECT

       indicator_value

    FROM

      `patents-public-data.worldbank_wdi.wdi_2016`

    WHERE

      (country_code = 'KOR') 

      AND indicator_name LIKE 'GDP per capita (current US$)'

      AND year < 2014

    ORDER BY year

        """



bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")



pd.options.display.max_rows=1000



client = bigquery.Client()

df = client.query(QUERY).to_dataframe()

df1 = client.query(INDICATORS).to_dataframe()

df_Obtained_Targets = client.query(TRAINTARGET).to_dataframe()



df2 = pd.merge(df, df1)

obtained_data = df2.values

obtained_targets = df_Obtained_Targets.values



arr = np.zeros((54, 90))

indicators = np.array([])



for i in obtained_data:

    n_row = i[0] -1960

    b_insert = True

    for j in range(len(indicators)):

        if i[1] == indicators[j]:

            arr[n_row, j] = i[2]

            b_insert = False

    if (b_insert):

        indicators = np.append(indicators, i[1])

        arr[n_row, len(indicators)-1] = i[2]

#print('indicators shape', indicators.shape)

for i in range(len(indicators)):

    if(indicators[i] == 'Merchandise exports to low- and middle-income economies in East Asia & Pacific (% of total merchandise exports)'):

        arr[1][i] = round((arr[0][i] + arr[2][i])/2, 15)

        

    if(indicators[i] == 'Merchandise imports from low- and middle-income economies in Sub-Saharan Africa (% of total merchandise imports)'):

        arr[4][i] = round((arr[3][i] + arr[5][i])/2, 15)

        

    if(indicators[i] == 'Merchandise exports to low- and middle-income economies outside region (% of total merchandise exports)'):

        arr[1][i] = round((arr[0][i] + arr[2][i])/2, 15)



train_data = np.empty((0, 90))

train_targets = np.array([])

test_data = np.empty((0, 90))

test_targets = np.array([])





for i in range(len(arr)):

    if i < 49:

        train_data = np.concatenate((train_data,np.expand_dims(arr[i], axis = 0)))

        train_targets = np.append(train_targets, obtained_targets[i], axis = 0)

    else:

        test_data = np.concatenate((test_data,np.expand_dims(arr[i], axis = 0)))

        test_targets = np.append(test_targets, obtained_targets[i], axis = 0)



        



        

        

             

        

mean = train_data.mean(axis = 0)

train_data -= mean

std = train_data.std(axis=0)

train_data /= std



test_data -= mean

test_data /= std



def build_model():

    model = models.Sequential()

    model.add(layers.Dense(1024, activation='relu', input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model





k = 4        





num_val_samples = len(train_data) // k

num_epochs = 300

all_scores = []

all_mae_histories = []

all_loss_histories = []

for i in range(k):

    print ('Fold #', i)

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]

    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]



    partial_train_data = np.concatenate(

        [train_data[:i * num_val_samples],

         train_data[(i + 1) * num_val_samples:]],

        axis = 0)

    partial_train_targets = np.concatenate(

        [train_targets[:i * num_val_samples],

         train_targets[(i + 1) * num_val_samples:]],

        axis = 0)



    model = build_model()

    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs = num_epochs, batch_size= 1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)

    all_scores.append(val_mae)

    mae_history = history.history['val_mean_absolute_error']

    loss_history = history.history['val_loss']

    all_mae_histories.append(mae_history)

    all_loss_histories.append(loss_history)



average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]







def smooth_curve (points, factor = 0.9):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points



smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(10, len(smooth_mae_history) + 10), smooth_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')



plt.figure()

smooth_loss_history = smooth_curve(average_loss_history[10:])

plt.plot(range(10, len(smooth_loss_history) + 10), smooth_loss_history)

plt.xlabel('Epochs')

plt.ylabel('Validation Loss')



plt.show()



min = average_mae_history[0]

min_index = 0

for i in range(80, len(average_mae_history)):

    if (min > average_mae_history[i]):

        min = average_mae_history[i]

        min_index = i



print('Minimum in mae:', min_index, min)

print('***FINISH***')

#1024, 64, 1 - epoch: 120~130 min:518 mean:590