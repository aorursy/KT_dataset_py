%matplotlib inline
from fastai.structured import *
from fastai.column_data import *
# np.set_printoptions(threshold=50, edgeitems=20)
PATH = '../input'
os.listdir(PATH)
# To reproduce the value in the next time
manual_seed = 555
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.deterministic = True
train_df_raw = pd.read_csv(f'{PATH}/train.csv', nrows=5000000)
test_df_raw = pd.read_csv(f'{PATH}/test.csv')
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
# this function will also be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

BB = (-74.5, -72.8, 40.5, 41.8)
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
    return dataset
def data_preprocessing(df, testset=0):
    add_travel_vector_features(df)
    if testset==0:
        df = df.dropna(how='any',axis='rows')
        df = df[(df.abs_diff_longitude<5) & (df.abs_diff_latitude<5)]
        df = df[df.fare_amount>0]
        df = df[(df.passenger_count >= 0) & (df.passenger_count <= 6)]
        df = df[select_within_boundingbox(df, BB)]
        
    df[['date','time','timezone']] = df['pickup_datetime'].str.split(expand=True)
    add_datepart(df, "date", drop=False)

    df[['hour','minute','second']] = df['time'].str.split(':',expand=True).astype('int64')
    df[['trash', 'order_no']] = df['key'].str.split('.',expand=True)
    df['order_no'] = df['order_no'].astype('int64')
    df = df.drop(['timezone','time', 'pickup_datetime','trash','date'], axis = 1)
    
    df = add_airport_dist(df)
    df['distance'] = sphere_dist(df['pickup_latitude'], df['pickup_longitude'], 
                                   df['dropoff_latitude'] , df['dropoff_longitude'])
    return df
train_df = data_preprocessing(train_df_raw)
test_df = data_preprocessing(test_df_raw, testset=1)
train_df = train_df.reset_index()
test_df = test_df.reset_index()
train_df.columns

cat_vars = ['passenger_count', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
    'Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start','hour','minute','second','order_no']

contin_vars = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
               ,'jfk_dist','ewr_dist','lga_dist','distance']

dep = 'fare_amount'
n = len(train_df); n
train_df = train_df[cat_vars+contin_vars+ [dep,'key']].copy()
test_df[dep] = 0
test_df = test_df[cat_vars+contin_vars+ [dep,'key']].copy()
for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()
apply_cats(test_df, train_df)
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')
train_df = train_df.set_index("key")
df, y, nas, mapper = proc_df(train_df, 'fare_amount', do_scale=True)
test_df = test_df.set_index("key")
df_test, _, nas, mapper = proc_df(test_df, 'fare_amount', do_scale=True,
                                  mapper=mapper, na_dict=nas)
# train_ratio = 0.75
train_ratio = 0.8
train_size = int(n * train_ratio); train_size
val_idx = list(range(train_size, len(df)))
y
def rmse(y_pred, targ):
    pct_var = (targ - y_pred)
    return math.sqrt((pct_var**2).mean())
# ,test_df=df_test
df_test
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars, bs=256,test_df=df_test)
cat_vars
cat_sz = [(c, len(train_df[c].cat.categories)+1) for c in cat_vars]
y
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz];emb_szs
max_y = np.max(y)
y_range = (0, max_y*1.2)
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
!ls ../input
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [256, 128, 64, 32, 8], [0.008,0.008, 0.008, 0.01, 0.01], y_range=y_range,tmp_name=TMP_PATH,models_name=MODEL_PATH)
m.lr_find()
m.sched.plot()
m.sched.plot_lr()
lr = 2e-5

m.fit(lr, 3, metrics=[rmse])
m.fit(lr, 6, cycle_len=1, metrics=[rmse])
m.fit(lr, 4, cycle_len=1, cycle_mult=2, metrics=[rmse])
m.fit(lr, 4, cycle_len=1, cycle_mult=2, metrics=[rmse])
pred_test=m.predict()
len(pred_test)
len(y[val_idx])
y[:20]
y_test = m.predict(True)
y_test = y_test.reshape(-1)

submission = pd.DataFrame(
    {'key': test_df.index, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
test_df.index