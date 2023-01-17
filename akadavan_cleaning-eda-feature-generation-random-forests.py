

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading useful libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import rc

import matplotlib.pyplot as plt

import matplotlib as mpl



from cycler import cycler



get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')



get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('dark_background')

mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#ff0000', '#0000ff',   '#00ffff','#ffA300', '#00ff00', 

     '#ff00ff', '#990000', '#009999', '#999900', '#009900', '#009999'])



rc('font', size=16)

rc('font',**{'family':'serif','serif':['Computer Modern']})

rc('text', usetex=False)

rc('figure', figsize=(12, 10))

rc('axes', linewidth=.5)

rc('lines', linewidth=1.75)
df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")
df.head()
df['scheduled_day'] = pd.to_datetime(df['ScheduledDay'], format= "%Y-%m-%d %H:%M:%S")

df['appointment_day'] = pd.to_datetime(df['AppointmentDay'], format= "%Y-%m-%d %H:%M:%S")
df.head()
df['no_show'] = df['No-show'].map({"No": 0, "Yes": 1}).astype(bool)

df.drop(columns=['No-show', "ScheduledDay","AppointmentDay"], inplace=True)
df.head()
df.rename(columns={'Hipertension': "hypertension", "Handcap": "handicap",

                       "PatientId": "patient_id", "AppointmentID": "appointment_id"}, inplace=True)

rename_cols = {col: col.lower() for col in df.columns}

df.rename(columns=rename_cols, inplace=True)

df['gender'] = df['gender'].map({"M": 0, "F": 1}).astype(bool)

for col in ['scholarship',"hypertension","diabetes","alcoholism","handicap","sms_received"]:

    df[col] = df[col].astype(bool)    
df.head()
df.dtypes
df.patient_id.nunique()
df.shape
pids_unique = df.patient_id.unique()

map_pids = {i: k for k, i in enumerate(pids_unique)}

df['patient_id'] = df['patient_id'].map(map_pids)
df.age.agg([max, min, 'mean'])
df.loc[df.age>90].shape
df['age'] = df['age'].clip(0,90)
df['appointment_day'].nunique()
df['appointment_day'].agg([min,max])
df['scheduled_day'].agg([min,max])
df['scheduled_day'].nunique()
(df['appointment_day'] < df['scheduled_day']).sum()
def diagnose_cols(df_small):

    for col in df.columns:

        nunique = df_small[col].nunique()

        print(f"number of unique values in {col}", nunique)

        if nunique == 1:

            print(df_small[col].unique())

    print("Average number of no-shows", df_small["no_show"].mean())
diagnose_cols(df.loc[df.appointment_day < df.scheduled_day])
df['no_show'].mean()
df.loc[df.appointment_day < df.scheduled_day,["scheduled_day","appointment_day"]].head()
df.loc[df.appointment_day < df.scheduled_day, "appointment_day"] = df.loc[

        df.appointment_day < df.scheduled_day, "appointment_day"] + pd.Timedelta("1d")
df.query('appointment_day < scheduled_day').shape
df.drop(df.loc[df.appointment_day < df.scheduled_day].index, inplace=True)
for col in ['patient_id', 'appointment_id']:

    df[col] = df[col].astype('category')
age_bins = {k: i for i, k in enumerate(np.sort(pd.cut(df['age'], bins= 9, retbins=False).unique()))}

df['age_group'] = pd.cut(df['age'], bins=9).map(age_bins).astype(int)
df.head()
# Plot no show for each age group, across the two genders

ax = df.pivot_table(values='no_show', index="age_group", columns="gender", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and gender");
ax = df.pivot_table(values='no_show', index="age_group", columns="scholarship", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and scholarship");
df.groupby("scholarship")['no_show'].mean()
ax = df.pivot_table(values='no_show', index="age_group", columns="hypertension", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and hypertension");
ax = df.pivot_table(values='no_show', index="age_group", columns="diabetes", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and diabetes");
ax = df.pivot_table(values='no_show', index="age_group", columns="alcoholism", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and alcoholism");
df.groupby('alcoholism')['no_show'].mean()
ax =df.pivot_table(values='no_show', index="age_group", columns="handicap", aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and handicap");
df.groupby('handicap')['no_show'].mean()
ax = df.pivot_table(index='age_group', columns='sms_received', values='no_show', aggfunc=np.mean).plot()

ax.set_xlabel("age groups");

ax.set_ylabel("Rate of no shows");

ax.set_title("Rate of no shows across age groups and sms_received");
df_rates = pd.DataFrame()

for col in df.dtypes[df.dtypes == 'bool'].index:

    if col != "no_show":

        mean_rate = df.groupby(col)['no_show'].mean()

        mean_rate.name = mean_rate.index.name

        #print(ser.name)

        mean_rate.index.name = ''

        df_rates[mean_rate.name] = mean_rate
df_rates = df_rates.T
ax = df_rates.plot.barh();

ax.set_title('average rate of no shows for different categories of no show variables');
df_agg = pd.DataFrame(index = np.array([(False, False), (False, True), (True, False), (True, True)]))

bool_cols =  df.dtypes[df.dtypes == 'bool'].index

bool_cols = bool_cols[bool_cols != 'no_show']



from itertools import combinations

for t in combinations(bool_cols, 2):

    cols_to_groupby = list(t)

    grouped = df.groupby(cols_to_groupby)['no_show'].mean()

    name1 = grouped.index.get_level_values(0).name

    name2 = grouped.index.get_level_values(1).name

    print(name1, name2)

    grouped.index = grouped.index.values

    col_name = f'{name1}_{name2}'

    df_agg[col_name] = grouped

df_agg = df_agg.T
df_agg.head()
df_agg.plot.barh(figsize=(10,30));
df.neighbourhood.nunique()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(df.groupby('neighbourhood')['no_show'].agg(['mean','count']).sort_values(by='count',

                                                                                  ascending=True))
df_geo = pd.read_csv("/kaggle/input/latitude-longitude-brazil/cleaned_geo_info.csv")
df_geo.head()
df_r = df_geo[['latitude','longitude']].round()

for cols in df_r.columns:

    print(df_r[cols].unique())
df_geo['long_round'] = df_geo.longitude.round()
# we remove the extreme values in longitude to make plotting easier. These extreme values correspond to off shore islands, which have very few

# appointments anyway

df_geo.drop(df_geo.query('long_round in [-35,-29]').index, inplace=True)
df_geo.plot.scatter(x = 'longitude', y='latitude')
df_neighbourhood = df.groupby('neighbourhood')['no_show'].mean().reset_index()
df_geo = pd.merge(df_geo, df_neighbourhood, how='left', on='neighbourhood')
df_geo.drop(columns='long_round', inplace=True)
df_geo.head()
plt.figure(figsize=(15,15))

plt.scatter(x=df_geo['longitude'], y=df_geo['latitude'], c=df_geo['no_show']);

plt.title('Geographic variation in no show rates');

plt.colorbar()
# %load utils.py

import pandas as pd

import numpy as np



def load_data():

    df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")

    # setting the correct date time formats

    df['scheduled_day'] = pd.to_datetime(df['ScheduledDay'], format= "%Y-%m-%d %H:%M:%S")

    df['appointment_day'] = pd.to_datetime(df['AppointmentDay'], format= "%Y-%m-%d %H:%M:%S")

    df.rename(columns={'Hipertension': "hypertension", "Handcap": "handicap",

                       "PatientId": "patient_id", "AppointmentID": "appointment_id"}, inplace=True)

    df['no_show'] = df['No-show'].map({"No": 0, "Yes": 1}).astype(bool)

    df.drop(columns=['No-show', "ScheduledDay","AppointmentDay"], inplace=True)

    rename_cols = {col: col.lower() for col in df.columns}

    df.rename(columns=rename_cols, inplace=True)

    df['gender'] = df['gender'].map({"M": 0, "F": 1}).astype(bool)

    for col in ['scholarship',"hypertension","diabetes","alcoholism","handicap","sms_received"]:

        df[col] = df[col].astype(bool)

    # patient_ids are float numbers. Mapping them to integers

    pids_unique = df.patient_id.unique()

    map_pids = {i: k for k, i in enumerate(pids_unique)}

    df['patient_id'] = df['patient_id'].map(map_pids)

    # Age more than 90 is unlikely, so clipping

    df.loc[df.age > 90, "age"] = 90

    # One age value is negative. Replacing with zero

    df.loc[df.age < 0, "age"] = 0

    # calling a function to remove errors in dates

    correct_date_errors(df)

    for col in ['patient_id', 'appointment_id']:

        df[col] = df[col].astype('category')

    # converting age into a categorical feature. This is useful for visualization and to create cross category features

    map_bins = {k: i for i, k in enumerate(np.sort(pd.cut(df['age'], bins= 9, retbins=False).unique()))}

    df['age_group'] = pd.cut(df['age'], bins=9).map(map_bins).astype(int)

    return df





# This function removes errors in date. Some appointment dates are behind scheduled dates. This is impossible.

# So I add 1 day to fix this. The ones that are still behind ( I counted five), I drop, since they are too few to affect prediction

def correct_date_errors(df):

    df['error_occured'] = df['appointment_day'] < df['scheduled_day']

    df.loc[df.appointment_day < df.scheduled_day, "appointment_day"] = df.loc[

        df.appointment_day < df.scheduled_day, "appointment_day"] + pd.Timedelta("1d")

    df.drop(df.loc[df.appointment_day < df.scheduled_day].index, inplace=True)
from itertools import combinations

import numpy as np

from tqdm.notebook import tqdm

from time import perf_counter

def get_features():

    time_now = perf_counter()

    df = load_data()

    # loading latitude and longitude data: pre-generated

    geo = pd.read_csv("/kaggle/input/latitude-longitude-brazil/cleaned_geo_info.csv")

    # merge the longitude latitude data

    df = pd.merge(df, geo, on = "neighbourhood", how="left")

    # select only boolean columns

    bool_cols =  df.dtypes[df.dtypes == 'bool'].index

    bool_cols = bool_cols[bool_cols != 'no_show']



    

    # make cross category boolean columns.

    # example: gender = F and alcoholism = True

    for combs in combinations(bool_cols,2):

        comb_list = list(combs)

        df['_'.join(comb_list)] = df[comb_list].product(axis=1)

    # interaction term between age and boolean columns

    for bool_col in bool_cols:

        df[f"{bool_col}_age"] = df["age"] * df[bool_col]

    df[bool_cols] = df[bool_cols].astype(np.int16)

    

    # neighbourhood are string objects. Convert them to integers

    unique_neighbourhoods = df.neighbourhood.unique()

    neighbourhood_map = {x: k for k,x in enumerate(unique_neighbourhoods)}

    df["neighbourhood"] = df["neighbourhood"].map(neighbourhood_map)

    # interaction feature between neighbourhoods and boolean columns

    for bool_col in bool_cols:

        df[f"{bool_col}_neighbourhoods"] = df["neighbourhood"] * df[bool_col]

    # interaction between neighbourhood and age

    #df["neighbourhood_age"]  = df["neighbourhood"]*10 + df["age_group"]

    #unique_pairs_map = {x:i for i,x in enumerate(df["neighbourhood_age"].unique())}

    #df["neighbourhood_age"] = df["neighbourhood_age"].map(unique_pairs_map)

    #df["neighbourhood_age"] = df["neighbourhood_age"].astype(int)

    df["neighbourhood"] = df["neighbourhood"].astype(int)

    

    df["appointment_date"] = df["appointment_day"].dt.date

    df["scheduled_date"] = df["scheduled_day"].dt.date

    # date features

    for date_type in ["appointment_day", "scheduled_day"]:

        df[f"{date_type}_day"] = df[date_type].dt.day 

        df[f"{date_type}_weekday"] = df[date_type].dt.weekday

        df[f"{date_type}_month"] = df[date_type].dt.month

        df[f"{date_type}_hour"] = df[date_type].dt.hour

    

    # time between appointments

    df["days_to_appointment"] = (df["appointment_day"] - df["scheduled_day"]).dt.days

    

    # time-based features

    # historic features for patient, location

    

    # patient:

    # for each patient and appointment: take scheduled_day

    # take all appointments of patient with appointmentday less than scheduled_day

    # take mean of df["no_show"]

    # similarly for locations

    add_history_to_df(df, "patient_id")

    add_history_to_df(df, "neighbourhood")

    df.drop(columns=["appointment_id", "patient_id", "age_group", 

                     "appointment_day", "appointment_day_hour", "scheduled_day", "error_occured", "appointment_date", "scheduled_date"], inplace = True)

    print(f"{df.shape[1]-1} features generated in ", perf_counter() - time_now, " seconds")

    return df





# This function calculates the historical cancellation probability for patient_id and location

# For example: at this location what is the probabiity of cancelled appointments till now?

def add_history_to_df(df, col_name):

    appointment_days = np.sort(df["appointment_day"].dt.date.unique())

    df[f"{col_name}_history"]  = 0

    print(f"generating {col_name} features")

    df_res = get_no_show_history(df, appointment_days, col_name)

    for k in range(1,len(appointment_days)):

        low_lim = appointment_days[k]

        try:

            up_lim = appointment_days[k+1]

        except:

            up_lim = appointment_days.max() + pd.Timedelta("2d")

        map_df = df_res.query("appointment_day == @low_lim").set_index(col_name)["no_show"]

        df_temp = df.query("@low_lim <= scheduled_date < @up_lim")

        df.loc[(df.scheduled_date >= low_lim) & (df.scheduled_date < up_lim),f"{col_name}_history"] = df_temp[col_name].map(map_df)

        

        

def get_no_show_history(df, appointment_days, col_name):

    u_pid = df[col_name].unique()

    df_full = []

    for k in appointment_days[1:]:

        df_past = df.query("appointment_date < @k")

        df_past = df_past.groupby(col_name)["no_show"].mean()

        df_all = pd.Series(0,index = u_pid, name="no_show")

        df_all.index.name = col_name

        df_all.update(df_past)

        df_all = df_all.reset_index()

        df_all["appointment_day"] = k

        df_full.append(df_all)

    df_res = pd.concat(df_full)

    return df_res
df_features = get_features()
predictors = df_features.columns[df_features.columns != 'no_show']

df_x = df_features[predictors]

df_y = df_features['no_show']
from sklearn.model_selection import train_test_split

# notice the stratification. This is important because of slightly imbalanced classes

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, stratify = df_y, random_state=23)
from sklearn.ensemble import RandomForestClassifier

rf  = RandomForestClassifier(n_jobs=-1, n_estimators=300)
rf.fit(x_train, y_train)
pd.Series(rf.feature_importances_, index = predictors).sort_values(ascending=False).head(20).plot.barh(figsize = (10,20))
# let us take 25 most important variables

most_imp_variables = pd.Series(rf.feature_importances_, index = predictors).sort_values(ascending=False).head(25).index
print(most_imp_variables)
from sklearn.metrics import roc_auc_score

rf  = RandomForestClassifier(n_jobs=-1, n_estimators=300, random_state=23)

rf.fit(x_train[most_imp_variables], y_train)

train_score = rf.predict_proba(x_train[most_imp_variables])[:,1]

print("ROC AUC curve for train data is ", roc_auc_score(y_train, train_score))

test_score = rf.predict_proba(x_test[most_imp_variables])[:,1]

print("ROC AUC curve for test data is" , roc_auc_score(y_test, test_score))
from sklearn.metrics import accuracy_score

test_pred = rf.predict(x_test[most_imp_variables])

accuracy_score(y_test, test_pred)