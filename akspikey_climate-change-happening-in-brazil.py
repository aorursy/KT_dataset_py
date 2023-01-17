import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
belem_data = pd.read_csv('../input/temperature-timeseries-for-some-brazilian-cities/station_belem.csv')
belem_data.columns
belem_data.head()
def process_data(data, state):

    # Loading the dataset

    state_data = pd.read_csv('../input/temperature-timeseries-for-some-brazilian-cities/{}.csv'.format(data))

    

    # Looping through the columns of the dataset

    for i in range(state_data.shape[0]):

        for col in state_data.columns:

            # checks whether it's missing data or not

            if state_data.iloc[i][col] == 999.90:

                # calculating the mean value and replacing it

                mean_val = state_data.iloc[i-1][col]+state_data.iloc[i-2][col]+state_data.iloc[i-3][col]

                state_data.at[i, col] = mean_val/3

    

    # calculating the mean value (metANN)

    state_data['mean'] = state_data.apply(lambda x: x[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']].mean(), axis=1)

    state_data['state'] = state

    return state_data
belem_data = process_data('station_belem', 'Belem')
belem_data.head()
fig, ax = plt.subplots(1,1)

fig.set_size_inches(20, 10)

sns.lineplot(x="mean", y="YEAR", data=belem_data)
curitiba_data = process_data('station_curitiba', 'Curitiba')

fortaleza_data = process_data('station_fortaleza', 'Fortaleza')

goiania_data = process_data('station_goiania', 'Goiania')

macapa_data = process_data('station_macapa', 'Macapa')

manaus_data = process_data('station_manaus', 'Manaus')
weather_data = {

    'belem_data': belem_data,

    'curitiba_data': curitiba_data,

    'fortaleza_data': fortaleza_data, 

    'goiania_data': goiania_data, 

    'macapa_data': macapa_data,

    'manaus_data': manaus_data

}
fig, ax = plt.subplots(6,4)



fig.set_size_inches(20, 20)



for index, key in enumerate(weather_data.keys()):

    ax[index, 0].plot("YEAR", "D-J-F", data=weather_data[key])

    ax[index, 1].plot("YEAR", "M-A-M", data=weather_data[key])

    ax[index, 2].plot("YEAR", "J-J-A", data=weather_data[key])

    ax[index, 3].plot("YEAR", "S-O-N", data=weather_data[key])
month_data = belem_data.drop(['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN', 'mean', 'state'], axis=1)

melted_data = month_data.melt(id_vars=["YEAR"], 

        var_name="Month", 

        value_name="Value")
fig, ax = plt.subplots(1,1)

fig.set_size_inches(25, 8)

ax = sns.heatmap(melted_data.pivot("Month","YEAR", "Value"),cmap="YlOrRd")