import numpy as np

import pandas as pd

from string import punctuation

import plotly.graph_objects as go

import re
train_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
train_df.head()
# print yard per offensive formation

fig = go.Figure(go.Bar(

            x=train_df.groupby('OffenseFormation').mean()['Yards'].values,

            y=train_df.groupby('OffenseFormation').mean()['Yards'].index,

            orientation='h', ))



fig.update_layout(title="Avg Yard Per Play For Different Formation", xaxis_title="Avg Yards", yaxis_title="Formation")



fig.show()
# one hot encoding

train_df = pd.concat([train_df, pd.get_dummies(train_df['OffenseFormation'], prefix='Formation')], axis=1)

# drop one variable

train_df = train_df.drop(['Formation_EMPTY'], axis=1)
', '.join(train_df['StadiumType'].value_counts().index)
# fixing the typos in stadium

def clean_StadiumType(txt):

    if pd.isna(txt):

        return np.nan

    txt = txt.lower() 

    txt = ''.join([c for c in txt if c not in punctuation])

    txt = re.sub('-', ' ', txt)

    txt = ' '.join(txt.split()) # remove additional whitespace

    txt = txt.strip()

    txt = txt.replace('outside', 'outdoor')

    txt = txt.replace('outdor', 'outdoor')

    txt = txt.replace('outddors', 'outdoor')

    txt = txt.replace('outdoors', 'outdoor')

    txt = txt.replace('oudoor', 'outdoor')

    txt = txt.replace('indoors', 'indoor')

    txt = txt.replace('ourdoor', 'outdoor')

    txt = txt.replace('retractable', 'rtr')

    txt = txt.replace('retr', 'rtr') 

    txt = txt.replace('roofopen', 'roof open')

    txt = txt.replace('roofclosed', 'roof closed')

    txt = txt.replace('closed dome', 'dome closed')

    return txt
train_df['StadiumType'] = train_df['StadiumType'].apply(clean_StadiumType)
fig = go.Figure(data=[go.Pie(labels=train_df['StadiumType'].value_counts().index, values=train_df['StadiumType'].value_counts().values)])

fig.show()
train_df['StadiumType'].unique()
# Stadium names for where StadiumType is null

train_df[train_df['StadiumType'].isnull()]['Stadium'].value_counts()
def get_roofOpen(StadiumType, weather):

    

    roof_open = {'outdoor': 1, 'open': 1, 'indoor open roof': 1, 'outdoor rtr roof open': 1, 

                 'rtr roof open': 1, 'heinz field': 1, 'cloudy': 1, 'bowl': 1, 'domed open': 1,

                 'indoor': 0, 'rtr roof closed': 0, 'indoor roof closed': 0, 'dome closed': 0, 

                 'dome': 0, 'domed closed': 0, 'domed': 0}

    

    if StadiumType:

        # if stadium type is set look for it in the dict above

        if roof_open.get(StadiumType):

            return roof_open.get(StadiumType)

        # if 'rtr roof' then decide based on the weather

        else:

            if weather == 'Rainy':

                return 0

            else:

                return 1

    # if Stadium Type is empty then we know it is one of the open air stadiums

    else:

        return 1
train_df['RoofOpen'] = train_df.apply(lambda row: get_roofOpen(row.StadiumType, row.GameWeather), axis=1)
roof_open_performance = train_df.groupby(['RoofOpen']).mean()['Yards']

print('Avg Yard Per Play With Roof Open: {0:1.3f} '.format(roof_open_performance[1]))

print('Avg Yard Per Play With Roof Closed: {0:1.3f} '.format(roof_open_performance[0]))
', '.join(train_df['Turf'].unique())
# copied from the discussion https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112681#latest-649087



def is_grass(Turf):

    

    grass = {'Field Turf': 0, 'A-Turf Titan': 0, 'Grass': 1, 'UBU Sports Speed S5-M': 0, 

            'Artificial': 0, 'DD GrassMaster': 0, 'Natural Grass': 1, 'UBU Speed Series-S5-M': 0, 

            'FieldTurf': 0, 'FieldTurf 360': 0, 'Natural grass': 1, 'grass': 1, 

            'Natural': 1, 'Artifical': 0, 'FieldTurf360': 0, 'Naturall Grass': 1, 'Field turf': 0, 

            'SISGrass': 0, 'Twenty-Four/Seven Turf': 0, 'natural grass': 1} 

    

    return grass.get(Turf)
train_df['Grass'] = train_df['Turf'].apply(is_grass)
turf_performance = train_df.groupby(['Grass']).mean()['Yards']

print('Avg Yard Per Play Without Grass: {0:1.3f} '.format(turf_performance[0]))

print('Avg Yard Per Play With Grass: {0:1.3f} '.format(turf_performance[1]))