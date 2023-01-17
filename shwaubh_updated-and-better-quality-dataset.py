import pandas as pd
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')
data.head()
data.rename(columns={'Unnamed: 0':'id'}, inplace=True)

data.set_index('id', inplace=True)

data.head()
floor = data['floor']

floor.replace('-',0, inplace=True)

data.head()
animal = data['animal']

animal.replace({'acept':1, 'not acept':0}, inplace = True)

data.head()
furnished = data['furniture']

furnished.replace({'furnished':1, 'not furnished':0}, inplace = True)

data.head()
hoa = data['hoa']

hoa = hoa.str[2:].apply(lambda x : x.replace(',',''))

#Some data are 'm info' and 'cluso'

i_index = hoa[hoa.str.contains('m')].index

c_index = hoa[hoa.str.contains('c')].index
hoa[i_index] = 0

hoa[c_index] = 0

data['hoa'] = hoa
data['hoa'].astype('int64')

data.head()
rent = data['rent amount']

rent = rent.str[2:].apply(lambda x : x.replace(',',''))

#Some data are 'm info' and 'cluso'

i_index = rent[rent.str.contains('m')].index

c_index = rent[rent.str.contains('c')].index
rent[i_index] = 0

rent[c_index] = 0

data['rent amount'] = rent

data['rent amount'].astype('int64')

data.head()
prop = data['property tax']

prop = prop.str[2:].apply(lambda x : x.replace(',',''))

#Some data are 'm info' and 'cluso'

i_index = prop[prop.str.contains('m')].index

c_index = prop[prop.str.contains('c')].index
prop[i_index] = 0

prop[c_index] = 0

data['property tax'] = prop

data['property tax'].astype('int64')

data.head()
fire = data['fire insurance']

fire = fire.str[2:].apply(lambda x : x.replace(',',''))

#Some data are 'm info' and 'cluso'

i_index = fire[fire.str.contains('m')].index

c_index = fire[fire.str.contains('c')].index
fire[i_index] = 0

fire[c_index] = 0

data['fire insurance'] = fire

data['fire insurance'].astype('int64')

data.head()
total = data['total']

total = total.str[2:].apply(lambda x : x.replace(',',''))

#Some data are 'm info' and 'cluso'

i_index = total[total.str.contains('m')].index

c_index = total[total.str.contains('c')].index
total[i_index] = 0

total[c_index] = 0

data['total'] = total

data['total'].astype('int64')

data.head()
new_csv = data.to_csv('updated_brasilian_housing_to_rent.csv')