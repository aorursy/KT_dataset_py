# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

import copy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def transformar(caminho, caminhoSaida):

    data = pd.read_csv(caminho, index_col=False)

    new_data = copy.deepcopy(data.iloc[1:, :]).reset_index()

    new_data['Lat_t1'] = data[' Lat'].iloc[:-1]

    new_data['Lon_t1'] = data[' Lon'].iloc[:-1]



    new_data['varLat'] = new_data['Lat_t1'] - new_data[' Lat']

    new_data['varLon'] = new_data['Lon_t1'] - new_data[' Lon']



    new_data[' Lat'] = [math.radians(v) for v in new_data[' Lat']]

    new_data['Lat_t1'] = [math.radians(v) for v in new_data['Lat_t1']]

    new_data['varLat'] = [math.radians(v) for v in new_data['varLat']]

    new_data['varLon'] = [math.radians(v) for v in new_data['varLon']]



    R = 6371 * (10**3)



    new_data['a'] = [((math.sin(v[1]['varLat']/2))**2) + (math.cos(v[1][' Lat']) * math.cos(v[1]['Lat_t1']) * (math.sin(v[1]['varLon']/2))**2) for v in new_data.iterrows()]



    new_data['c'] = [2 * math.atan2(math.sqrt(v[1]['a']), math.sqrt(1-v[1]['a'])) for v in new_data.iterrows()]



    new_data['distancia'] = R * new_data['c']    

    

    new_data.to_csv(caminhoSaida)

    

    return new_data
for dirname, _, filenames in os.walk('/kaggle/input/vascofisico'):

    for filename in filenames:

        print('Transformando', os.path.join(dirname, filename))

        if filename.endswith('.csv'):

            transformar('/kaggle/input/vascofisico/' + filename,filename)                    