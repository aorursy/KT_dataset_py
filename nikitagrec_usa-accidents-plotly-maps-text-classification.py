import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')
# create df for state accidents

import plotly.graph_objects as go

state_count_acc = pd.value_counts(data['State'])



fig = go.Figure(data=go.Choropleth(

    locations=state_count_acc.index,

    z = state_count_acc.values.astype(float),

    locationmode = 'USA-states',

    colorscale = 'Reds',

    colorbar_title = "Count Accidents",

))



fig.update_layout(

    title_text = '2016 - 2019 US Traffic Accident Dataset by State',

    geo_scope='usa',

)



fig.show()
!pip install plotly-geo
df_county = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/laucnty16.csv')

df_county.head(3)
df_county['county_full'] = df_county['County Name/State Abbreviation'].apply(lambda x: x.split(', ')[0])

df_county['county_name'] = df_county['county_full'].apply(lambda x: x.split(' County')[0])



county_count_acc = pd.value_counts(data['County'])

fips_county_df = df_county[['county_name', 'County FIPS Code', 'State FIPS Code']].merge(county_count_acc, left_on='county_name', right_index=True)
import plotly.figure_factory as ff



fips_county_df['State FIPS Code'] = fips_county_df['State FIPS Code'].apply(lambda x: str(x).zfill(2))

fips_county_df['County FIPS Code'] = fips_county_df['County FIPS Code'].apply(lambda x: str(x).zfill(3))

fips_county_df['FIPS'] = fips_county_df['State FIPS Code'] + fips_county_df['County FIPS Code']



colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",

    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",

    "#08519c", "#0b4083", "#08306b"

]

endpts = list(np.linspace(1,30000, len(colorscale) - 1))

fips = fips_county_df['FIPS'].tolist()

values = fips_county_df['County'].tolist()





fig = ff.create_choropleth(

    fips=fips, values=values, scope=['usa'],

    binning_endpoints=endpts, colorscale=colorscale,

    show_state_data=False,

    show_hover=True,

    asp = 2.9,

    title_text = 'USA County accidents count',

    legend_title = 'Accidents count'

)

fig.layout.template = None

fig.show()
data_sever = data.sample(n=10000)



fig = go.Figure(data=go.Scattergeo(

        locationmode = 'USA-states',

        lon = data_sever['Start_Lng'],

        lat = data_sever['Start_Lat'],

        text = data_sever['City'],

        mode = 'markers',

        marker = dict(

            size = 8,

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = 'Reds',

            cmin = data_sever['Severity'].max(),

        color = data_sever['Severity'],

        cmax = 1,

            colorbar_title="Severity"

        )))



fig.update_layout(

        title = 'Severity of accidents',

        geo = dict(

            scope='usa',

            projection_type='albers usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.7,

            subunitwidth = 0.7

        ),

    )

fig.show()
county_severity_acc = data[['County','Severity']].groupby('County').mean().reset_index()

fips_county_sev = df_county[['county_name', 'County FIPS Code', 'State FIPS Code']].merge(county_severity_acc, left_on='county_name', right_on='County')

fips_county_sev['Severity'] = fips_county_sev['Severity'].apply(lambda x: round(x,1))
fips_county_sev['State FIPS Code'] = fips_county_sev['State FIPS Code'].apply(lambda x: str(x).zfill(2))

fips_county_sev['County FIPS Code'] = fips_county_sev['County FIPS Code'].apply(lambda x: str(x).zfill(3))

fips_county_sev['FIPS'] = fips_county_sev['State FIPS Code'] + fips_county_sev['County FIPS Code']





fips = fips_county_sev['FIPS'].tolist()

values = fips_county_sev['Severity'].tolist()





fig = ff.create_choropleth(

    fips=fips, values=values, scope=['usa'],

#     binning_endpoints=endpts,

#     show_state_data=False,

#     show_hover=True,

#     asp = 2.9,

    title_text = 'USA accidents severity (mean)',

    legend_title = 'Accidents severity'

)

fig.layout.template = None

fig.show()
data_sever = data.sample(n=10000)[['Start_Lng','Start_Lat','City','Visibility(mi)','Severity']]

data_sever.dropna(inplace=True)



fig = go.Figure(data=go.Scattergeo(

        locationmode = 'USA-states',

        lon = data_sever['Start_Lng'],

        lat = data_sever['Start_Lat'],

        text = data_sever['City'],

        mode = 'markers',

        marker = dict(

            size = data_sever['Visibility(mi)'],

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = 'Blues',

            cmin = data_sever['Severity'].max(),

        color = data_sever['Severity'],

        cmax = 1,

            colorbar_title="Severity"

        )))



fig.update_layout(

        title = 'Severity & Visibility of accidents',

        geo = dict(

            scope='usa',

            projection_type='albers usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.7,

            subunitwidth = 0.7

        ),

    )

fig.show()
data_sever = data.sample(n=15000)[['Start_Lng','Start_Lat','City','Distance(mi)','Severity']]

data_sever.dropna(inplace=True)



fig = go.Figure(data=go.Scattergeo(

        locationmode = 'USA-states',

        lon = data_sever['Start_Lng'],

        lat = data_sever['Start_Lat'],

        text = data_sever['City'],

        mode = 'markers',

        marker = dict(

            size = data_sever['Distance(mi)'],

            opacity = 0.8,

            reversescale = True,

            autocolorscale = False,

            symbol = 'circle',

            line = dict(

                width=1,

                color='rgba(102, 102, 102)'

            ),

            colorscale = 'Viridis',

            cmin = data_sever['Severity'].max(),

        color = data_sever['Severity'],

        cmax = 1,

            colorbar_title="Severity"

        )))



fig.update_layout(

        title = 'Severity & Distance of accidents',

        geo = dict(

            scope='usa',

            projection_type='albers usa',

            showland = True,

            landcolor = "rgb(250, 250, 250)",

            subunitcolor = "rgb(217, 217, 217)",

            countrycolor = "rgb(217, 217, 217)",

            countrywidth = 0.7,

            subunitwidth = 0.7

        ),

    )

fig.show()
from more_itertools import sliced

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.callbacks.callbacks import EarlyStopping
text_data = data[['Description','Severity']].sample(n=1000, random_state=10)

text_data.head()
for i in range(15):

    print(text_data['Description'].iloc[i])
# The maximum number of words to be used. (most frequent)

MAX_NB_WORDS = 100

# Max number of words in each complaint.

MAX_SEQUENCE_LENGTH = 500

# This is fixed.

EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(text_data['Description'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(text_data['Description'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', X.shape)



Y = pd.get_dummies(text_data['Severity']).values

print('Shape of label tensor:', Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# choose epochs and batch_size

epochs = 15

batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)])
accr = model.evaluate(X_test,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Loss')

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show();