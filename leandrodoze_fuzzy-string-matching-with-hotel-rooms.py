# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/room_type.csv')
df.count()
df.head(10)
from fuzzywuzzy import fuzz
fuzz.ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')
fuzz.ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')
fuzz.ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')
fuzz.partial_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')
fuzz.partial_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')
fuzz.partial_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')
fuzz.token_sort_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')
fuzz.token_sort_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')
fuzz.token_sort_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')
fuzz.token_set_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')
fuzz.token_set_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')
fuzz.token_set_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')
def get_ratio(row):
    name1 = row['Expedia']
    name2 = row['Booking.com']
    return fuzz.token_set_ratio(name1, name2)

rated = df.apply(get_ratio, axis=1)
rated.head(10)
greater_than_70_percent = df[rated > 70]
greater_than_70_percent.count()
greater_than_70_percent.head(10)
len(greater_than_70_percent) / len(df)
greater_than_70_percent = df[rated > 60]
greater_than_70_percent.count()
len(greater_than_70_percent) / len(df)