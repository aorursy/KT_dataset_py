import pandas as p

import keras as k

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



import matplotlib.pyplot as plt



%matplotlib inline
dtype_list = { 'Registry Number' : 'object', 'Entity of Record Reg Number' : 'object', 'Zip' : 'category', \

       'Entity Type' : 'category', 'Associated Name Type' : 'category', 'State' : 'category', \

       'City' : 'category' }

only_cols = ['Zip', 'City', 'State', 'Entity Type', 'Associated Name Type' ]
df = p.read_csv('../input/Active_Businesses_LLC.csv', dtype=dtype_list, parse_dates=True, usecols=only_cols, )



df.head()
def label_encode(series):

    label_encoder = LabelEncoder()

    

    return label_encoder.fit_transform(series)
df['Zip'].fillna(method='backfill', inplace=True)
new_df = p.DataFrame()



new_df['Zip'] = label_encode(df['Zip'])

new_df['City'] = label_encode(df['City'])

new_df['State'] = label_encode(df['State'])

new_df['Entity Type'] = label_encode(df['Entity Type'])

new_df['Associated Name Type'] = label_encode(df['Associated Name Type'])



new_df.head()
plt.subplot(1, 3, 1)



plt.scatter(new_df['Zip'], new_df['City'], 3)

plt.xlabel('Zip Code')

plt.ylabel('City')



plt.subplot(1, 3, 2)

plt.hist(new_df['Zip'])

plt.xlabel('Zip')



plt.subplot(1, 3, 3)

plt.hist(new_df['City'])

plt.xlabel('City')



plt.show()
plt.subplot(1, 2, 1)

plt.hist(new_df['Associated Name Type'], 10)

plt.xlabel('Name Type')



plt.subplot(1, 2, 2)

plt.hist(new_df['Entity Type'])

plt.xlabel('Entity Type')



plt.show()
plt.figure(figsize=(12,12))



plt.subplot(1, 2, 1)



plt.hist2d(new_df['Zip'], new_df['Associated Name Type'])

plt.xlabel('Zip Code')

plt.ylabel('Associated Name Type')



plt.subplot(1, 2, 2)



plt.hist2d(new_df['City'], new_df['Associated Name Type'])

plt.xlabel('City')



plt.show()