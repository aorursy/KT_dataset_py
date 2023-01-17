import pandas as pd

import numpy as np
filepath = '/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv'



WORKDAYS = '1111110'

HOLIDAYS = ['2020-03-08','2020-03-25', '2020-03-30', '2020-03-31']



GMT8_OFFSET = 3600 * 8

DURATION_1DAY = 3600 * 24



def mat_to_dict(mat):

    n = len(mat)

    return {i*n+j: mat[i][j] for i in range(n) for j in range(n)}



sla_matrix_1st_attempt = [

    [3, 5, 7, 7],

    [5, 5, 7, 7],

    [7, 7, 7, 7],

    [7, 7, 7, 7],

]

sla_matrix_2nd_attempt = [

    [3, 3, 3, 3],

    [3, 3, 3, 3],

    [3, 3, 3, 3],

    [3, 3, 3, 3],

]

locations = ["Metro Manila", "Luzon", "Visayas", "Mindanao"]

locations = [loc.lower() for loc in locations]

min_length = min(map(len, locations))

trunc_location_to_index = {loc[-min_length:]: i for i, loc in enumerate(locations)}
%%time

dtype = {

    'orderid': np.int64,

    'pick': np.int64,

    '1st_deliver_attempt': np.int64,

    '2nd_deliver_attempt': np.float64,

    'buyeraddress': np.object,

    'selleraddress': np.object,

}

df = pd.read_csv(filepath, dtype=dtype)
%%time

# convert address to index

df['buyeraddress'] = df['buyeraddress'].apply(lambda s: s[-min_length:].lower()).map(trunc_location_to_index)

df['selleraddress'] = df['selleraddress'].apply(lambda s: s[-min_length:].lower()).map(trunc_location_to_index)
%%time

# convert unix datetime(seconds)stamps to unix datetime(date)stamps

dt_columns = ['pick', '1st_deliver_attempt', '2nd_deliver_attempt']

df[dt_columns[-1]] = df['2nd_deliver_attempt'].fillna(0).astype(np.int64)

df[dt_columns] = (df[dt_columns] + GMT8_OFFSET) // DURATION_1DAY
%%time

# compute number of working days between time intervals

t1 = df['pick'].values.astype('datetime64[D]')

t2 = df['1st_deliver_attempt'].values.astype('datetime64[D]')

t3 = df['2nd_deliver_attempt'].values.astype('datetime64[D]')

df['num_days1'] = np.busday_count(t1, t2, weekmask=WORKDAYS, holidays=HOLIDAYS)

df['num_days2'] = np.busday_count(t2, t3, weekmask=WORKDAYS, holidays=HOLIDAYS)
%%time

# compute sla based on addresses

to_from = df['buyeraddress']*4 + df['selleraddress']

df['sla1'] = to_from.map(mat_to_dict(sla_matrix_1st_attempt))

df['sla2'] = to_from.map(mat_to_dict(sla_matrix_2nd_attempt))
%%time

# compute if deliver is late

df['is_late'] = (df['num_days1'] > df['sla1']) | (df['num_days2'] > df['sla2'])

df['is_late'] = df['is_late'].astype(int)
%%time

# write to file

df[['orderid', 'is_late']].to_csv('submission.csv', index=False)