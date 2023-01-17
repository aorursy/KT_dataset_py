# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.patches as patches

from IPython.core.display import display, HTML



from plotly import tools, subplots

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_file

from bokeh.io import output_notebook

from sklearn.impute import SimpleImputer



from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import preprocessing

import lightgbm as lgb

import plotly.express as px

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype





import warnings

warnings.filterwarnings('ignore')



py.init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os,gc

import seaborn as sns



path="/kaggle/input/ashrae-energy-prediction/"



train=pd.read_csv(path+'train.csv')

test=pd.read_csv(path+'test.csv')



# Any results you write to the current directory are saved as output.
train['timestamp']=pd.to_datetime(train["timestamp"], format='%Y-%m-%d %H:%M:%S')

test['timestamp']=pd.to_datetime(test["timestamp"], format='%Y-%m-%d %H:%M:%S')
train.info()
train.head(2)
nouniquebuild=train['building_id'].nunique()

buildingallmeterlist=[]



count4=0

count1=0



for build in train['building_id'].unique():

    metertypes=train[train['building_id']==build]['meter'].unique()

    if len(metertypes)==4:

        buildingallmeterlist.append(build)

        count4+=1

    elif len(metertypes)==1:

        count1+=1



display(HTML(f"""<br>Number of buildings in the dataset: {nouniquebuild:,}</br>

             <br>No of buildings having all meter types: {count4:,}</br>

             <br>No of buildings having only one meter type: {count1:,}</br>

             """))
def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train.isna().sum()
def _generate_bar_plot_hor(df, col, title, color, w=None, h=None, lm=0):

    cnt_srs = df[col].value_counts().sort_values(ascending=False)

    cnt_srs.index=["Electricity", "ChilledWater", "Steam", "HotWater"]

    

    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], orientation = 'h',

        marker=dict(color=color))



    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)

    data = [trace]

    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
_generate_bar_plot_hor(train,'meter', "Distribution of meter reading", '#1E90FF', 600, 400)
sample=train[train['building_id'].isin(buildingallmeterlist)]
gapminder = px.data.gapminder().query("continent=='Oceania'")

fig = px.line(sample, x="timestamp", y="meter_reading", color='building_id')

fig.show()
gapminder = px.data.gapminder().query("continent=='Oceania'")

fig = px.line(sample[sample['building_id']==1249], x="timestamp", y="meter_reading", color='meter')

fig.show()
output_notebook()

def make_plot(title, hist, edges, xlabel):

    p = figure(title=title, tools='', background_fill_color="#fafafa")

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],

           fill_color="#f975ae", line_color="white", alpha=0.5)



    p.y_range.start = 0

    p.xaxis.axis_label = f'Log of {xlabel} meter reading'

    p.yaxis.axis_label = 'Probability'

    p.grid.grid_line_color="white"

    return p



temp_df = train[train["meter"]==0]

hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)

p1 = make_plot("Meter Reading Distribution for Electricity meter", hist, edges, "electricity")



temp_df = train[train["meter"]==1]

hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)

p2 = make_plot("Meter Reading Distribution for Chilled Water meter", hist, edges, 'chill water')



temp_df = train[train["meter"]==2]

hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)

p3 = make_plot("Meter Reading Distribution for Steam meter", hist, edges, 'steam')



temp_df = train[train["meter"]==3]

hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)

p4 = make_plot("Meter Reading Distribution for Hot Water meter", hist, edges, 'hot water')



show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=400))



weather_train=pd.read_csv(path+'weather_train.csv')

weather_test=pd.read_csv(path+'weather_test.csv')



weather_train['timestamp']=pd.to_datetime(weather_train["timestamp"], format='%Y-%m-%d %H:%M:%S')

weather_test['timestamp']=pd.to_datetime(weather_test["timestamp"], format='%Y-%m-%d %H:%M:%S')


def Nanper(df):

    percentdf=pd.DataFrame(columns=['Columnname','Percentage'])



    for ix,col in enumerate(df.columns):

        percentdf.loc[ix,'Percentage']=len(df[df[col].isnull()])/len(df)*100

        percentdf.loc[ix,'Columnname']=col

        

    return percentdf
Nanper(weather_train)
weather = pd.concat([weather_train,weather_test],ignore_index=True)

weather_key = ['site_id', 'timestamp']

temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

del weather
# calculate ranks of hourly temperatures within date/site_id chunks

temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')



# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)

df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)



# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.

site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

site_ids_offsets.index.name = 'site_id'



def timestamp_align(df):

    df['offset'] = df.site_id.map(site_ids_offsets)

    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))

    df['timestamp'] = df['timestamp_aligned']

    del df['timestamp_aligned']

    return df

weather_train_df = timestamp_align(weather_train)

weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))



weather_test_df = timestamp_align(weather_test)

weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
#checking the percentage now

Nanper(weather_train_df)
del weather_train,weather_test,temp_skeleton,df_2d
buildingdata=pd.read_csv(path+'building_metadata.csv')
Nanper(buildingdata)
cnt_srs = buildingdata["year_built"].value_counts()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="#1E90FF",

    ),

)



layout = go.Layout(

    title=go.layout.Title(

        text="Year built - Count",

        x=0.5

    ),

    font=dict(size=14),

    width=1000,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Year built")
cnt_srs = buildingdata["floor_count"].value_counts()



trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="#1E90FF",

    ),

)



layout = go.Layout(

    title=go.layout.Title(

        text="Floors - Count",

        x=0.5

    ),

    font=dict(size=14),

    width=1000,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="Floor")
traindfv1=train.merge(buildingdata,how='left',on=['building_id'])

traindfv2=traindfv1.merge(weather_train_df,how='left',on=['site_id','timestamp'])



del traindfv1,weather_train_df,train



traindfv2=reduce_mem_usage(traindfv2)
#Filling the missing values of year built and floor count with maximum occurance values

floor_fill=traindfv2['year_built'].value_counts().head(1).keys()[0]

year_fill=traindfv2['floor_count'].value_counts().head(1).keys()[0]



traindfv2['year_built'].fillna(year_fill,inplace=True)

traindfv2['floor_count'].fillna(floor_fill,inplace=True)
testdfv1=test.merge(buildingdata,how='left',on=['building_id'])

del test,buildingdata



testdfv2=testdfv1.merge(weather_test_df,how='left',on=['site_id','timestamp'])

del testdfv1,weather_test_df



testdfv2=reduce_mem_usage(testdfv2)
testdfv2['year_built'].fillna(year_fill,inplace=True)

testdfv2['floor_count'].fillna(floor_fill,inplace=True)
Nanper(traindfv2)
traindfv2['hour'] = traindfv2['timestamp'].dt.hour

traindfv2['day'] = traindfv2['timestamp'].dt.day

traindfv2['weekday'] = traindfv2['timestamp'].dt.weekday

traindfv2['month'] = traindfv2['timestamp'].dt.month

traindfv2['year'] = traindfv2['timestamp'].dt.year



testdfv2['hour'] = testdfv2['timestamp'].dt.hour

testdfv2['day'] = testdfv2['timestamp'].dt.day

testdfv2['weekday'] = testdfv2['timestamp'].dt.weekday

testdfv2['month'] = testdfv2['timestamp'].dt.month

testdfv2['year'] = testdfv2['timestamp'].dt.year



del traindfv2["timestamp"], testdfv2['timestamp']
traindfv2.head(1)
traindfv2['square_feet'] = np.log1p(traindfv2['square_feet'].values)

traindfv2['meter_reading'] = np.log1p(traindfv2['meter_reading'].values)



testdfv2['square_feet'] = np.log1p(testdfv2['square_feet'].values)
#testdfv2,testNAlist=reduce_mem_usage(testdfv2)
le = preprocessing.LabelEncoder()



traindfv2['primary_use'] = le.fit_transform(traindfv2['primary_use'])

testdfv2['primary_use'] = le.transform(testdfv2['primary_use'])
features=[ 'meter','primary_use', 'square_feet', 'year_built', 'floor_count',

       'air_temperature', 'cloud_coverage', 'dew_temperature',

       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

       'wind_speed', 'hour', 'day', 'weekday', 'month', 'year']



target='meter_reading'


folds = 4

seed = 42

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)



models = []



## stratify data by building_id

for tr_idx, val_idx in tqdm(kf.split(traindfv2, traindfv2['building_id']), total=folds):

    

    def fit_regressor(tr_idx, val_idx):

        tr_x, tr_y = traindfv2[features].iloc[tr_idx], traindfv2[target][tr_idx]

        vl_x, vl_y = traindfv2[features].iloc[val_idx], traindfv2[target][val_idx]

        print({'train size':len(tr_x), 'eval size':len(vl_x)})



        tr_data = lgb.Dataset(tr_x, label=tr_y)

        vl_data = lgb.Dataset(vl_x, label=vl_y)  

        

        clf = lgb.LGBMRegressor(n_estimators=1000,

                                learning_rate=0.28,

                                feature_fraction=0.9,

                                subsample=0.2,  # batches of 20% of the data

                                subsample_freq=1,

                                num_leaves=20,

                                metric='rmse')

        clf.fit(tr_x, tr_y,

                eval_set=[(vl_x, vl_y)],

                early_stopping_rounds=50,

                verbose=200)



        return clf

    clf = fit_regressor(tr_idx, val_idx)

    models.append(clf)

    

gc.collect()
feature_importance = np.mean([m._Booster.feature_importance(importance_type='gain') for m in models], axis=0)

sorted(zip(feature_importance, features), reverse=True)
# split test data into batches

set_size = len(testdfv2)

iterations = 50

batch_size = set_size // iterations



print(set_size, iterations, batch_size)

assert set_size == iterations * batch_size



del traindfv2
meter_reading = []



for i in tqdm(range(iterations)):

    pos = i*batch_size

    fold_preds = [np.expm1(model.predict(testdfv2[features].iloc[pos : pos+batch_size])) for model in models]

    meter_reading.extend(np.mean(fold_preds, axis=0))



print(len(meter_reading))

assert len(meter_reading) == set_size
submission = pd.read_csv(f'{path}/sample_submission.csv')

submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero
submission.to_csv('submission.csv', index=False)