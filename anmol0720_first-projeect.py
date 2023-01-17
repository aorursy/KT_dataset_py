import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
d1=pd.read_csv('../input/calendar.csv')

d1=d1.iloc[:3818,:2]

d2=pd.read_csv('../input/listings.csv')

d2=d2.iloc[:3818,:]

d3=pd.read_csv('../input/reviews.csv')

d3=d3.iloc[:3818,:]
d=pd.concat([d1,d2,d3],axis=1)

d.head()
d['monthly_price'] = pd.to_numeric(d['monthly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')

d['price'] = pd.to_numeric(d['price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')

d['weekly_price'] = pd.to_numeric(d['weekly_price'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')

d['security_deposit'] = pd.to_numeric(d['security_deposit'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')

d['cleaning_fee'] = pd.to_numeric(d['cleaning_fee'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')

d['extra_people'] = pd.to_numeric(d['extra_people'].apply(lambda x: str(x).replace('$', '').replace(',', '')),errors='coerce')
d=d._get_numeric_data()

d=d.drop(['license','square_feet'],axis=1)

d=d.dropna()

d