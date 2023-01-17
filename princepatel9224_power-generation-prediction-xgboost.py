# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import train_test_split
plant_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant_df.head()
plant_df.shape
plant_df_columns = plant_df.columns.tolist()

plant_df[plant_df_columns].isnull().sum()
plant_df['DATE_TIME']= pd.to_datetime(plant_df['DATE_TIME'],format='%d-%m-%Y %H:%M') 
SOURCE_KEY_list = plant_df['SOURCE_KEY'].unique()

day_of_month_list = plant_df['DATE_TIME'].dt.day.unique()

month_list = plant_df['DATE_TIME'].dt.month.unique()
def data_collection():

    main_df = pd.DataFrame()

    for i in day_of_month_list:

        for j in month_list:

            df=plant_df[(plant_df.DATE_TIME.dt.month == j) & (plant_df.DATE_TIME.dt.day == i) ][-len(SOURCE_KEY_list):]

            df = df.drop(['PLANT_ID', 'DC_POWER', 'AC_POWER', 'TOTAL_YIELD'],axis = 1)

            df = df[df.DAILY_YIELD != 0]

            main_df = main_df.append(df, ignore_index=True)

    return main_df

main_df = data_collection()
main_df.index = main_df.DATE_TIME.dt.date.astype("datetime64[ns]")

main_df = main_df.drop(["DATE_TIME"],axis=1)
for i in SOURCE_KEY_list:

    df = main_df[main_df.SOURCE_KEY == i]

    df.DAILY_YIELD.plot()

    plt.title("SOURCE_KEY : %s"%i)

    plt.show()


for i in SOURCE_KEY_list:

    df = main_df[main_df.SOURCE_KEY == i]

    df.DAILY_YIELD.plot()

plt.show()
Fault_SOURCE_KEY_list=  ["McdE0feGgRqW7Ca","bvBOhCH3iADSZry","sjndEbLyjtCKgGv","wCURE6d3bPkepu2"]
#remove data who having Fault_SOURCE_KEY_list in main_df

for i in Fault_SOURCE_KEY_list:

    main_df = main_df[main_df.SOURCE_KEY != i]
Unfault_SOURCE_KEY = main_df.SOURCE_KEY.unique()

for i in Unfault_SOURCE_KEY:

    df = main_df[main_df.SOURCE_KEY == i]

    df.DAILY_YIELD.plot()

plt.show
main_df['dayofweek'] = main_df.index.dayofweek

main_df['quarter'] = main_df.index.quarter

main_df['month'] = main_df.index.month

main_df['year'] = main_df.index.year

main_df['dayofyear'] = main_df.index.dayofyear

main_df['dayofmonth'] = main_df.index.day

main_df['weekofyear'] = main_df.index.weekofyear
label_encoder = preprocessing.LabelEncoder() 

main_df['SOURCE_KEY']= label_encoder.fit_transform(main_df['SOURCE_KEY']) 

X = main_df[['SOURCE_KEY','dayofweek', 'quarter','month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']]

y = main_df["DAILY_YIELD"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
reg = xgb.XGBRegressor(n_estimators=500,

                       objective ='reg:squarederror',

                       learning_rate = 0.16,

                       colsample_bytree=0.6,

                       max_depth = 5,

                       min_child_weight = 6)

reg.fit(X_train, y_train,

        eval_set=[(X_train, y_train), (X_test, y_test)],

        early_stopping_rounds=50)
plot_importance(reg, height=0.9)
y_pred = reg.predict(X_test)

mean_squared_error(y_test,y_pred,squared=False)
# n = number of days to predict future generation

def create_df(n):

    prediction_df = pd.DataFrame()

    for i in range(0,n):

        df = pd.DataFrame()

        df["SOURCE_KEY"] = Unfault_SOURCE_KEY

        df["DATE_TIME"] = "2020-06-%d"%(i+15)

        prediction_df = prediction_df.append(df)

    prediction_df['DATE_TIME']= pd.to_datetime(prediction_df['DATE_TIME']) 

    prediction_df.index = prediction_df.DATE_TIME.dt.date.astype("datetime64[ns]")

    prediction_df = prediction_df.drop(["DATE_TIME"],axis=1)

    prediction_df['dayofweek'] = prediction_df.index.dayofweek

    prediction_df['quarter'] = prediction_df.index.quarter

    prediction_df['month'] = prediction_df.index.month

    prediction_df['year'] = prediction_df.index.year

    prediction_df['dayofyear'] = prediction_df.index.dayofyear

    prediction_df['dayofmonth'] = prediction_df.index.day

    prediction_df['weekofyear'] = prediction_df.index.week

    return prediction_df

        
x = create_df(2)

df_copy = x.copy() 

x['SOURCE_KEY'] = label_encoder.fit_transform(x['SOURCE_KEY'])

y_prediction = reg.predict(x)
df_copy["DAILY_YIELD_prediction"] = y_prediction
df_copy