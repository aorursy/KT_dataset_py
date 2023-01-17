import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_color_codes("muted")

import matplotlib.pyplot as plt

% matplotlib inline



numeric = "../input/train_numeric.csv"
features = pd.read_csv(numeric, nrows=1).drop(['Response', 'Id'], axis=1).columns.values



def orgainize(features):

    line_features = {}

    station_features = {}

    lines = set([f.split('_')[0] for f in features])

    stations = set([f.split('_')[1] for f in features])

    

    for l in lines:

        line_features[l] = [f for f in features if l+'_' in f]

        

    for s in stations:

        station_features[s] = [f for f in features if s+'_' in f]

        

            

    return line_features, station_features



line_features, station_features = orgainize(features)



print("Features in Station 32: {}".format( station_features['S32'] ))
station_error = []

for s in station_features:

    cols = ['Id', 'Response']

    cols.extend(station_features[s])

    df = pd.read_csv(numeric, usecols=cols).dropna(subset=station_features[s], how='all')

    error_rate = df[df.Response == 1].size / float(df[df.Response == 0].size)

    station_error.append([df.shape[1]-2, df.shape[0], error_rate]) 

    

station_data = pd.DataFrame(station_error, 

                         columns=['Features', 'Samples', 'Error_Rate'], 

                         index=station_features).sort_index()

station_data
plt.figure(figsize=(8, 20))

sns.barplot(x='Error_Rate', y=station_data.index.values, data=station_data, color="red")

plt.title('Error Rate between Production Stations')



plt.xlabel('Station Error Rate')

plt.show()
data = pd.read_csv(numeric, nrows=100)



def make_features(df):

    new_features = pd.DataFrame({})

    for s in station_features.keys():

        station_data = df[station_features[s]]

        col = s+'_max'

        new_features[col] = station_data.max(axis=1).fillna(-1.)

        col = s+'_min'

        new_features[col] = station_data.min(axis=1).fillna(-1.)

    return new_features



data = make_features(data)

data.head()