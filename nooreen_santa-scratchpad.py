import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine

gifts = pd.read_csv('../input/gifts.csv')

gifts.dtypes
sum(gifts['Weight'])/1000
len(gifts)
haversine((gifts.loc[0,'Latitude'],gifts.loc[0,'Longitude']), (gifts.loc[1,'Latitude'],gifts.loc[0,'Longitude'])) 
