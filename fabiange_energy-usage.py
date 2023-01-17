from fastai2.basics import *

from fastai2.tabular.all import *

from fastai2.callback.all import *
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = Path('/kaggle/input/ashrae-energy-prediction')
train = pd.read_csv(path/'train.csv', nrows=1000000)

bldg = pd.read_csv(path/'building_metadata.csv')

weather_train = pd.read_csv(path/"weather_train.csv")
len(train)
train = train[np.isfinite(train['meter_reading'])]
train = train.merge(bldg, left_on = 'building_id', right_on = 'building_id', how = 'left')
train.head()
train = train.merge(weather_train, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'])
del weather_train, bldg
train.head()
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"] = train["timestamp"].dt.day

train["weekend"] = train["timestamp"].dt.weekday

train["month"] = train["timestamp"].dt.month
train.head()
train.drop('timestamp', axis=1, inplace=True)

train['meter_reading'] = np.log1p(train['meter_reading'])
train.head()
cat_vars = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter"]

cont_vars = ["square_feet", "year_built", "air_temperature", "cloud_coverage",

              "dew_temperature"]

dep_var = 'meter_reading'
procs = [Normalize, Categorify, FillMissing]

splits = RandomSplitter()(range_of(train))
train = TabularPandas(train, procs, cat_vars, cont_vars, y_names=dep_var, splits=splits, block_y=RegressionBlock())
dls = train.dataloaders()
dls.show_batch()
with open(r"train.pkl", "wb") as output_file:

    pickle.dump(train, output_file)
emb_szs = get_emb_sz(train)
cont_len = len(train.cont_names); cont_len
net = TabularModel(emb_szs, cont_len, 1, [200,100])
net
learn = tabular_learner(dls, [200,100], loss_func=MSELossFlat(), metrics=accuracy, n_out=1)
learn.fit(1)
test = pd.read_csv(path/'test.csv')

bldg = pd.read_csv(path/'building_metadata.csv')

weather_test = pd.read_csv(path/"weather_test.csv")
test = test[np.isfinite(test['meter_reading'])]
test = test.merge(bldg, left_on = 'building_id', right_on = 'building_id', how = 'left')
test = test.merge(weather_test, left_on = ['site_id', 'timestamp'], right_on = ['site_id', 'timestamp'])
test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour

test["day"] = test["timestamp"].dt.day

test["weekend"] = test["timestamp"].dt.weekday

test["month"] = test["timestamp"].dt.month
test.drop('timestamp', axis=1, inplace=True)

test['meter_reading'] = np.log1p(test['meter_reading'])
#to_test = TabularPandas(test, procs, cat_vars, cont_vars, y_names=dep_var, block_y=RegressionBlock())
to_test = TabularPandas(test, procs, cat_vars, cont_vars, y_names=dep_var, block_y=RegressionBlock())

test_dl = TabDataLoader(to_test, bs=128, shuffle=False, drop_last=False)
preds, _ = learn.get_preds(dl=test_dl) 

preds = np.expm1(preds.numpy())
submission = pd.DataFrame(columns=['row_id', 'meter_reading'])
test.head()
submission['row_id'] = test['building_id']
submission['meter_reading'] = preds
submission.head()
submission.to_csv('submission.csv')
!ls
!pip install kaggle
!kaggle competitions submit -c ashrae-energy-prediction -f submission.csv -m "First try"