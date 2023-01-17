# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from geopy.geocoders import Nominatim



geolocator = Nominatim(user_agent="openstreetmap")
location = geolocator.geocode("Rua Camélia, Osasco, Sao Paulo")

print((location.latitude, location.longitude))
def get_lat_lon(x):

    try:

        location = geolocator.geocode(x)

        return (location.longitude, location.latitude)

    except: return (0, 0)
data = pd.read_csv("/kaggle/input/eleicoes-municipais-2016-sp/sao_paulo.csv", index_col=False)
data.head()
address_data = pd.DataFrame()

address_data["address"] = data["endereço"].unique()
address_data.shape
address_data["coordinates"] = address_data["address"].apply(lambda x: get_lat_lon(x))
(address_data["coordinates"] == (0, 0)).mean()
df = data.merge(address_data, left_on="endereço", right_on="address")
df["coordinates"]