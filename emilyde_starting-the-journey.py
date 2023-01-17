# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = {

'cal': pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv'),

'prc': pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv'),

'sal': pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

}

    

sam_sub_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

def describe_data(df):

    print("Data Types:")

    print(df.dtypes)

    print("Rows and Columns:")

    print(df.shape)

    print("Column Names:")

    print(df.columns)

    print("Null Values:")

    print(df.apply(lambda x: sum(x.isnull()) / len(df)))

    print("First 10 rows:")

    print(df.head(10))

describe_data(data['cal'])
describe_data(data['prc'])
describe_data(data['sal'])
#data['cal'][["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = data['cal'][["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")

#data['cal'][["wm_yr_wk", "year"]] = data['cal'][["wm_yr_wk", "year"]].astype("int16") 

#data['cal']["date"] = data['cal']["date"].astype("datetime64")

#data['cal'].dtypes
#nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

#for feature in nan_features:

#    data['cal'][feature].fillna('Empty', inplace = True)
#data['cal'][["d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = data['cal'][["d", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")
#print(data['cal']['event_type_1'].unique())

#print(data['cal']['event_name_1'].unique())

#print(data['cal']['event_type_2'].unique())

#print(data['cal']['event_name_2'].unique())
#data['cal'] = data['cal'].drop(['weekday'], axis=1)
#data['prc'][["store_id", "item_id"]] = data['prc'][["store_id","item_id"]].astype("category")

#data['prc'][["wm_yr_wk"]] = data['prc'][["wm_yr_wk"]].astype("int16")

#data['prc'][["sell_price"]] = data['prc'][["sell_price"]].astype("float16")
#data['sal'].loc[:, "d_1":] = data['sal'].loc[:, "d_1":].astype("int16")

#data['sal'][["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]] = data['sal'][["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].astype("category")
#combined_cal_prc_df = pd.merge(data['cal'], data['prc'], on='wm_yr_wk', how='inner')
#print(combined_cal_prc_df.dtypes)

#print(data['sal'].dtypes)
#combined_df = pd.merge(combined_cal_prc_df, data['sal'], on = ["store_id", "item_id"], copy = False, how='inner')
#Først ændrer vi kalender dataen, hvor datatyperne bliver omdannet til mindre bit værdier.

#data['cal'][["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = data['cal'][["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")

#data['cal'][["wm_yr_wk", "year"]] = data['cal'][["wm_yr_wk", "year"]].astype("int16")

#"date" bliver omdannet til Datetime så vi kan arbejde med det senere som timeseries.

#data['cal']["date"] = data['cal']["date"].astype("datetime64")



#Mange algoritmer ignorere eller melder fejl ved NaN værdier, derfor er det en god ide at omdanne til specifikke værdier.

#nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

#for feature in nan_features:

#    data['cal'][feature].fillna('unknown', inplace = True)



#objects omskrives til categories for at maskinen ikke selv skal parse dataen.

#data['cal'][["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = data['cal'][["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")
#Sal får også omdanet værdier. Her sørger loc for at køre alle kolonner igennem fra "d_1" og frem.

#data['sal'].loc[:, "d_1":] = data['sal'].loc[:, "d_1":].astype("int16")
#Her bruges loc til at ændre hver række. Der oprettes en "id" kollonne som er en sammensætning af "item_id" og "store_id" kolonnerne (plus en string). 

#Lidt ligesom "id" kolonnen i sales_train_validation

#data['prc'].loc[:, "id"] = data['prc'].loc[:, "item_id"] + "_" + data['prc'].loc[:, "store_id"] + "_validation"
#"item_id" bliver splittet i 2 colonner, som er opdelt vha. "_". Kolonnerne bliver indsat i price datasettet.

#data['prc'] = pd.concat([data['prc'], data['prc']["item_id"].str.split("_", expand=True)], axis=1)

#kolonne 0 og 1 (dem vi lige har oprettet) bliver renamet til noget nyt. Det stemmer overens med Sales_train_validation. 

#data['prc'] = data['prc'].rename(columns={0:"cat_id", 1:"dept_id"})

#Objekter laves til Category types

#data['prc'][["store_id", "item_id", "cat_id", "dept_id"]] = data['prc'][["store_id","item_id", "cat_id", "dept_id"]].astype("category")

#kolonne 2 bliver nu droppet.

#data['prc'] = data['prc'].drop(columns=2)
#def make_dataframe():

#    # Wide format dataset oprettet med data fra "sal"

#    #Først rengøres "sal" datasettet, hvor en række kolonner droppes. så kun "d" kolonnerne er tilbage (1913 kolonner).

#    df_wide_train = data['sal'].drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T

#    #Datasettet indexes efter "date" fra "cal" datasettet. En date kolonne bliver oprettet, hvor rækkerne fordeles op til række 1913 i date.

#    df_wide_train.index = data['cal']["date"][:1913]

#    #"id" rows fra "sal" datasettet bliver sat som kolonne navne, hvor rowsne er værdier fra "d kolonnerne".

#    df_wide_train.columns = data['sal']["id"]

#    

#    # oprettelse af test label dataset

#    # np.zeros returnerer et nyt array med shape på 56 med df_wide_train's kolonner.

#    # Derefter indesker den datoerne fra 'cal' indtil kolonnerne 1913, og fordels på kolonnerne fra df_wide_train datasættet. 

#    df_wide_test = pd.DataFrame(np.zeros(shape=(56, len(df_wide_train.columns))), index=data['cal'].date[1913:], columns=df_wide_train.columns)

#    #De 2 nye dataset sammensættes

#    df_wide = pd.concat([df_wide_train, df_wide_test])

#

#    # Wide-formatet konverteres til long-format.

#    # df_wide genindekseres fra kolonnerne, og omdannes til rækker. en ny kolonne bliver oprettet til long-formatet.

#    df_long = df_wide.stack().reset_index(1)

#    # de nye labels angives.

#    df_long.columns = ["id", "value"]

#

#    #Dataframes der ikke længere bruges, slettes (for perfomance, altså mindre RAM forbrug)

#    del df_wide_train, df_wide_test, df_wide

#    gc.collect() #Garbage Collection

#    

#    #df bliver oprettes, ud fra en merge på df_long og "cal" datasettet. reset_index sørger for at df starter fra hvor df_long er på 0 index.

#    #der merges på fælles værdier.

#    df = pd.merge(pd.merge(df_long.reset_index(), data['cal'], on="date"), data['prc'], on=["id", "wm_yr_wk"])

#    #alle "d" kolonner droppes, da vi har oprettet "values" kolonnen. Det fjerner redudant data.

#    df = df.drop(columns=["d"])

#df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")

#    #performance...

#    df["sell_price"] = df["sell_price"].astype("float16")   

#    df["value"] = df["value"].astype("int8")

#    df["state_id"] = df["store_id"].str[:2].astype("category")

#    df["id"] = df["id"].str[:2].astype("category")

#

#

#    del df_long

#    gc.collect()

#

#    return df

#

#df = make_dataframe()
#df.dtypes
#Opdelte date kolonner oprettes ud fra "date" kolonnen. Det er så man kan bruge det til Timeseries (f.eks. ARIMA).

#def add_date_feature(df):

#    df["year"] = df["date"].dt.year.astype("int16")

#    df["month"] = df["date"].dt.month.astype("int8")

#    df["week"] = df["date"].dt.week.astype("int8")

#    df["day"] = df["date"].dt.day.astype("int8")

#    df["quarter"]  = df["date"].dt.quarter.astype("int8")

#    return df

#

#df = add_date_feature(df)
#df.head()
#del data['cal'], data['prc'], data['sal']

#gc.collect()
#df.memory_usage(index = True) 
#describe_data(df)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def create_dataset(train_df, calendar_df, price_df, N=(365 * 2)):

    

    

    # Opret kolonner

    aggregate_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    series_columns = [f'd_{i}' for i in range(1, 1914)]

    test_series_columns = [f'd_{i}' for i in range(1914, 1942)]

    calendar_columns = [

        'date',

        'wm_yr_wk',

        'd',

        'event_name_1',

        'event_type_1',

        'event_name_2',

        'event_type_2',

        'snap_CA',

        'snap_TX',

        'snap_WI'

    ]

    calendar_events_columns = [

        'event_name_1',

        'event_type_1',

        'event_name_2',

        'event_type_2'

    ]



    # Indekser kolonner

    train_df_aggregate = train_df[aggregate_columns]



    # Hent sampled series

    if N is not None:

        sample_series_columns = series_columns[-N:]

    series_df = train_df[sample_series_columns]



    # Sample train df

    train_df = pd.concat([train_df_aggregate, series_df], axis=1)



    # Collapse train df

    train_df = (

        train_df

        .set_index(aggregate_columns)

        .stack().reset_index()

        .rename(

            columns={

                'level_6': 'd',

                0: 'sales'

            }

        )

    )



    # Oprettelse af test data

    test_array = np.zeros((train_df_aggregate.shape[0], len(test_series_columns)))



    test_df = pd.DataFrame(

        test_array,

        columns=test_series_columns

    )

    test_df = pd.concat([train_df_aggregate, test_df], axis=1)



    # Collapse test data

    test_df = test_df.set_index(aggregate_columns).stack().reset_index().rename(

        columns={'level_6': 'd', 0: 'sales'}

    )

    test_df['sales'] = np.nan

    train_df['datasetType'] = 1

    test_df['datasetType'] = 0



    # Sammensætning af train og test data

    data = pd.concat([train_df, test_df])



    # Forberedelse af calendar data kolonner

    calendar_df['date'] = pd.to_datetime(calendar_df['date'])

    calendar_df['event_name_1'] = calendar_df['event_name_1'].fillna('None')

    calendar_df['event_type_1'] = calendar_df['event_type_1'].fillna('None')

    calendar_df['event_name_2'] = calendar_df['event_name_1'].fillna('None')

    calendar_df['event_type_2'] = calendar_df['event_type_2'].fillna('None')

    calendar_df = calendar_df[calendar_columns]



    # Sammensætning af alt data

    data = pd.merge(data, calendar_df, on=['d'], how='left')

    data = pd.merge(data, price_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    data = data.drop(['d', 'wm_yr_wk'], axis=1)



    # Encode labels

    encoder_cols = aggregate_columns + calendar_events_columns

    for col in encoder_cols:

        if col == 'id':

            continue

        unique_values = sorted(data[col].unique().tolist())

        encoder_values = range(len(unique_values))

        label_encoder = dict(zip(unique_values, encoder_values))

        data[col] = data[col].map(label_encoder)



    data = data.sort_values(['id', 'date'])



    return reduce_mem_usage(data)
df = create_dataset(data['sal'], data['cal'],data['prc'])
describe_data(df)
import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")

corr = df.corr()



f, ax = plt.subplots(figsize=(11,9))



sns.heatmap(corr,square=True, linewidth=1.5, cmap="coolwarm")

ax.set_title('Correlation matrix visualiseret')
cat_plot_df = df.groupby(["cat_id", "date"])["sales"].sum()
plt.figure(figsize=(15,5))

plt.plot(cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 0].index.get_level_values("date"), cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 0].values, label="FOODS")

plt.plot(cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 1].index.get_level_values("date"), cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 1].values, label="HOBBIES")

plt.plot(cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 2].index.get_level_values("date"), cat_plot_df[cat_plot_df.index.get_level_values("cat_id") == 2].values, label="HOUSEHOLD")



plt.title("Sales by category")

plt.xlabel("Year")

plt.ylabel("sales")

plt.legend()
#cat_plot_df_start = df.groupby(["cat_id", "date"])["sales"].sum()
#cat_plot_df_start = cat_plot_df_start.loc[(cat_plot_df_start.index.get_level_values("date") >= "2011-01-01") & (cat_plot_df_start.index.get_level_values("date") <= "2012-12-31")]

#plt.figure(figsize=(15,5))

#plt.plot(cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")

#plt.plot(cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")

#plt.plot(cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), cat_plot_df_start[cat_plot_df_start.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")

#

#plt.title("Categories in the beginning of the set")

#plt.xlabel("Month")

#plt.ylabel("sales")

#plt.legend()
#cat_plot_df_2015 = df.groupby(["cat_id", "date"])["sales"].sum()
#cat_plot_df_2015 = cat_plot_df_2015.loc[(cat_plot_df_2015.index.get_level_values("date") >= "2015-01-01") & (cat_plot_df_2015.index.get_level_values("date") <= "2015-12-31")]

#plt.figure(figsize=(15,5))

#plt.plot(cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")

#plt.plot(cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")

#plt.plot(cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), cat_plot_df_2015[cat_plot_df_2015.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")

#

#plt.title("Categories 2015 (latest)")

#plt.xlabel("Month")

#plt.ylabel("sales")

#plt.legend()
#state_plot_df = df.groupby(["state_id", "date"])["sales"].sum()
#plt.figure(figsize=(15,5))

#plt.plot(state_plot_df[state_plot_df.index.get_level_values("state_id") == "CA"].index.get_level_values("date"), state_plot_df[state_plot_df.index.get_level_values("state_id") == "CA"].values, label="CA")

#plt.plot(state_plot_df[state_plot_df.index.get_level_values("state_id") == "TX"].index.get_level_values("date"), state_plot_df[state_plot_df.index.get_level_values("state_id") == "TX"].values, label="TX")

#plt.plot(state_plot_df[state_plot_df.index.get_level_values("state_id") == "WI"].index.get_level_values("date"), state_plot_df[state_plot_df.index.get_level_values("state_id") == "WI"].values, label="WI")

#

#plt.title("Sales by state")

#plt.xlabel("Year")

#plt.ylabel("sales")

#plt.legend()
#from statsmodels.tsa.stattools import adfuller
#x = df['sales'].values

#results = adfuller(x)

#print(results)
tiny_df = df[['date', 'sales']]

tiny_df= tiny_df.groupby('date', as_index=False).mean()

tiny_df = tiny_df.dropna()
describe_data(tiny_df)
y = tiny_df.set_index(['date'])

y.head()
y.plot(figsize=(20, 4))

plt.show()
y['2015-12'].plot(figsize=(20, 4))

plt.show()
from pylab import rcParams

import statsmodels.api as sm

import itertools

rcParams['figure.figsize'] = 18, 8

ds = y['2015-09']

ds.head()

decomposition = sm.tsa.seasonal_decompose(ds, model='additive')

fig = decomposition.plot()

plt.show()
p = d = q = range(0, 5)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#Her tjekker vi efter AIC værdier, for at finde den SARIMAX model så muligt.

#for param in pdq:

#    for param_seasonal in seasonal_pdq:

#        try:

#            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)

#            results = mod.fit()

#            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))

#        except: 

#            print('Jobcenter')
#ARIMA(1, 0, 0)x(1, 1, 0, 12)12 - AIC:54.6186669793878

#ARIMA(0, 0, 0)x(2, 4, 4, 12)12 - AIC:17.294407051803006

#ARIMA(0, 0, 1)x(1, 2, 1, 12)12 - AIC:15.760302266684189

#ARIMA(0, 0, 0)x(4, 4, 4, 12)12 - AIC:-3.5164340802543776

#ARIMA(0, 0, 0)x(2, 3, 3, 12)12 - AIC:18.13247436274471

#ARIMA(0, 0, 1)x(3, 0, 1, 12)12 - AIC:2.105 <-------------------- vi valgte denne, da den er tættest på 0



mod = sm.tsa.statespace.SARIMAX(y,

                                order=(0, 0, 1),

                                seasonal_order=(3, 0, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
#Vi bruger dette til at vurdere hvor præcis vores data er.

results.plot_diagnostics(figsize=(18, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2016-02-24'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2016':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(20, 4))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



ax.set_xlabel('Date')

ax.set_ylabel('Retail_sold')

plt.legend()

plt.show()
from sklearn.metrics import mean_squared_error as MSE

forecast = results.get_prediction(start=-28)

mean_forecast_is = forecast.predicted_mean



s_rmse_test = MSE(y[-28:], mean_forecast_is)**(1/2)

print('SARIMAX model RMSE: {:.2f}'.format(s_rmse_test))
pred_uc = results.get_forecast(steps=28)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 4))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()

plt.show()
forecast = pred_uc.predicted_mean

forecast.head(28)
from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



X= tiny_df[['date']]

y_r= tiny_df['sales']                                       



X_train, X_test, y_train, y_test = train_test_split(X, y_r, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor 

# Vi instantiere vores clasifier med 400 trees

rf = RandomForestRegressor(n_estimators=400,

random_state=42)



rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)





# tjekker root mean squared error

rmse_test = MSE(y_test, y_pred)**(1/2)

print('RandomForest model RMSE: {:.2f}'.format(rmse_test))
# Visualising the Random Forest Regression results 

X_grid = np.array(X, dtype=np.datetime64)  

  

# reshape for reshaping the data into a len(X_grid)*1 array,  

# i.e. to make a column out of the X_grid value                   

X_grid = X_grid.reshape((len(X_grid), 1)) 

  

# Scatter plot for original data 

plt.scatter(X_grid, rf.predict(X_grid), color = 'blue')   

  

# plot predicted data 

plt.plot(X, y_r, color = 'green')  

plt.title('Random Forest Regression') 

plt.xlabel('Position level') 

plt.ylabel('Salary') 

plt.show()
print('SARIMAX model RMSE: {:.2f}'.format(s_rmse_test))

print('RandomForest model RMSE: {:.2f}'.format(rmse_test))