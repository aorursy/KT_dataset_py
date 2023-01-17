import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from datetime import datetime
from datetime import timedelta

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import warnings
from pandas.core.common import SettingWithCopyWarning

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Upload main train data - UI weekly data - including all states
#https://oui.doleta.gov/unemploy/claims.asp
train_all = gpd.read_file("/kaggle/input/ui-claims/UIWeekly_all.csv").drop(['geometry'],axis=1)
train_all["Date"] = pd.to_datetime(train_all["Date"])
train_all = train_all.query('Province_State!="Puerto Rico" and Province_State!="Virgin Islands"')
train_all = train_all.query("Date>'2020-01-22'")
#Prepare a dataframe (train_df) with 51 states (from train_all) and daily frequency date from 2020-01-22 to 2020-05-04 to merge with daily frequency temporal data (i.e. Google trend, COVID quarantine etc.)
train_start = datetime.strptime("2020-01-22","%Y-%m-%d")
train_end = datetime.strptime("2020-05-04","%Y-%m-%d")
date_list = [train_start + timedelta(days=x) for x in range((train_end-train_start).days+1)]
Province_State = train_all.Province_State.unique()

train_list = []
for x in range(len(Province_State)):
    for y in range(len(date_list)):
        innerlist = [Province_State[x],date_list[y]]
        train_list.append(innerlist)
train_df = pd.DataFrame(train_list,columns=["Province_State","Date"])
#Upload Google Trend data and merge with train_df
#https://trends.google.com/trends/explore?date=today%203-m&geo=US-WY&q=file%20for%20unemployment
Google_trend_df = gpd.read_file("/kaggle/input/ui-claims/Google_Trend_v2.csv").drop(["geometry"],axis=1)
Google_trend_df["Date"] = pd.to_datetime(Google_trend_df["Date"])
Google_trend_df["Google_Trend"] = Google_trend_df["Google_Trend"].astype("float")
train_df = train_df.merge(Google_trend_df,
                          on=["Province_State","Date"],
                          how="left",
                          )
#Since Google Trend data starts from 2020-02-06 (90days before 2020-05-3), fill the data for date before 2/6 as 0 for Google_Trend
train_df.loc[(train_df["Date"]<"2020-02-06"),"Google_Trend"] = 0
train_df.head(5)
#Upload quarantine ("stay at home") and restriction ("public place") data
#https://www.usatoday.com/storytelling/coronavirus-reopening-america-map/
quarantine_data_df = gpd.read_file("/kaggle/input/ui-claims/Quarantine_v2.csv")
quarantine_data_df["SAH_start"] = pd.to_datetime(quarantine_data_df["SAH_start"],dayfirst=True,errors='coerce', format="%m/%d/%Y")
quarantine_data_df["SAH_end"] = pd.to_datetime(quarantine_data_df["SAH_end"])
quarantine_data_df["Restriction_Start"] = pd.to_datetime(quarantine_data_df["Restriction_Start"])
quarantine_data_df["Restriction_Easing"] = pd.to_datetime(quarantine_data_df["Restriction_Easing"])
#the use of cumsum() function will populate number 1 starting the date from quarantine data after merge.
def update_dates(a_df, col_update):
    """
    This creates a boolean time series with one after the start of confinements (different types : schools, restrictions or quarantine)
    """
    gpdf = a_df.groupby("Province_State")
    new_col = gpdf.apply(lambda df : df[col_update].notnull().cumsum()).reset_index(drop=True)
    a_df[col_update] = new_col

for col in ["SAH_start", "SAH_end", "Restriction_Start","Restriction_Easing"]:
    train_df = train_df.merge(quarantine_data_df[["Province_State", col]],
                          left_on=["Province_State", "Date"],
                          right_on=["Province_State", col],
                          how="left",
                          )
    update_dates(train_df, col)
#If quarantine ends or restriction being eased on Date A, fill 0 for the respective series from A onwards
train_df['quarantine']=[row.SAH_start if row.SAH_end==0 else 0 for idx,row in train_df.iterrows()]
train_df['restriction']=[row.Restriction_Start if row.Restriction_Easing==0 else 0 for idx,row in train_df.iterrows()]
train_df.tail(10)
#Upload UI 2019 Q4 baseline and Population for Age 18+ data (population data were found not improving the model) and merge with train_df
#https://oui.doleta.gov/unemploy/data_summary/DataSum.asp
#https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-detail.html
UI_base = gpd.read_file("/kaggle/input/ui-claims/UI_2019Q4_Base_v2.csv").drop(["geometry"],axis=1)
UI_base.rename({"Initial Claims": "UI_base"}, axis=1,inplace=True)
UI_base["UI_base"] = UI_base["UI_base"].astype("float")
UI_base["Pop_above_18"] = UI_base["Pop_above_18"].astype("float")
UI_base["Pop_above_18"] = UI_base["Pop_above_18"]/max(UI_base["Pop_above_18"])
train_df = train_df.merge(UI_base,
                          on=["Province_State"],
                          how="left",
                          )
#Upload GDP Industry data
gdp_industry_df = gpd.read_file("/kaggle/input/ui-claims/GDP_Industry_v2.csv").drop(["geometry"],axis=1)
Sector = ["Agriculture","Energy","Utilities","Construction","Durable","Nondurable","Wholesale","Retail","Transportation","Information","Finance","Real-estate",
"Technology","Management","Services","Education","Health-care","Entertainment","Food","Other","Gov"]
gdp_industry_df["Sum"] = gdp_industry_df["Sum"].astype("float")
for col in Sector:
    gdp_industry_df[f"{col}"] = gdp_industry_df[f"{col}"].astype("float")
    gdp_industry_df[f"{col}"] = gdp_industry_df[f"{col}"]/gdp_industry_df["Sum"]

train_df = train_df.merge(gdp_industry_df,
                          on=["Province_State"],
                          how="left",
                          )
#Upload UI daily data; UI daily data is used to enlarge the training dateset. That said, it also introduce data leakage. Hence UI daily data is only used for training not testing
#The Daily UI data were found for the following 6 states:
#DC:https://does.dc.gov/publication/unemployment-compensation-claims-data
#MN:https://mn.gov/deed/data/current-econ-highlights/ui-statistics.jsp
#MT:http://dli.mt.gov/labor-market-information
#NC:https://des.nc.gov/need-help/covid-19-information/unemployment-claims-data
#PA:https://www.uc.pa.gov/COVID-19/Pages/UC-Claim-Statistics.aspx
#WI:https://dwd.wisconsin.gov/covid19/public/ui-stats.htm
UI_daily_df = gpd.read_file("/kaggle/input/ui-claims/UIDaily.csv").drop(['field_4','geometry'],axis=1)
UI_daily_df["Date"] = pd.to_datetime(UI_daily_df["Date"])
UI_daily_df.rename({"Initial Claims": "UI"}, axis=1,inplace=True)
train_df_daily = UI_daily_df.merge(train_df,
                          on=["Province_State", "Date"],
                          how="left",
                          )
train_df_daily['UI'] = train_df_daily['UI'].astype("float")
train_df_daily = train_df_daily.query("Date>'2020-01-22'")
#checkpoint length of train_df_daily is 339
len(train_df_daily)
#Create temperal data with series of 20 days and shift by 1 day
days_in_sequence = 21

trend_list = []
demo_input = []

with tqdm(total=len(list(train_df_daily.Province_State.unique()))) as pbar:
    for province in train_df_daily.Province_State.unique():
        province_df = train_df_daily.query(f"Province_State=='{province}'")
        n = 0
        #for i in range(0,len(province_df),int(days_in_sequence/2)):
        for i in range(0,len(province_df),1):
            n += 1
            if i+days_in_sequence<=len(province_df):
                #prepare all the temporal inputs
                google_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Google_Trend.values]
                restriction_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].restriction.values]
                quarantine_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                #preparing all the demographic inputs
                demo_input = []
                for col in Sector:
                    col = float(province_df[f"{col}"].iloc[0])
                    demo_input.append(col)

                expected_trend = float(province_df.iloc[i+days_in_sequence-1].Google_Trend)
                #expected weekly claim is the sum of past 7 days including today
                expected_claims = 0
                for a in range(i+days_in_sequence-7, i+days_in_sequence):
                    expected_claims = expected_claims + float(province_df.iloc[a].UI)
                expected_claims_norm = expected_claims / float(province_df.iloc[i].UI_base) * 13

                trend_list.append({"google_trend":google_trend,
                                   "restriction_trend":restriction_trend,
                                   "quarantine_trend":quarantine_trend,
                                   "demographic_inputs":demo_input,
                                   "expected_trends":expected_trend,
                                   "expected_claims":expected_claims_norm})
        pbar.update(1)
trend_df_daily = pd.DataFrame(trend_list)
#After adjust the Daily UI in 2020 by 2019Q4 UI base, the variables are in the same scale, no further normalization is needed.
trend_df_daily.describe().round(2)
#Checkpoint number of UI daily data is 219
len(trend_df_daily)
#Prepare UI Weekly data for the remaining 45 states (51-6)
s_list = list(train_df_daily.Province_State.unique())+["Guam"]
query_df = train_df[~train_df["Province_State"].isin(s_list)]
len(query_df.Province_State.unique())
UI_weekly_df = train_all[~train_all["Province_State"].isin(s_list)]
len(UI_weekly_df.Province_State.unique())
train_df_weekly = query_df.merge(UI_weekly_df,
                          on=["Province_State", "Date"],
                          how="left",
                          )
train_df_weekly['Initial_Claims'] = train_df_weekly['Initial_Claims'].astype("float")
train_df_weekly['UI_norm'] = train_df_weekly['Initial_Claims']/(train_df_weekly['UI_base']/13)
train_df_weekly = train_df_weekly.query("Date>='2020-01-26' and Date<='2020-04-18'")
days_in_sequence = 21

trend_list = []
demo_input = []

with tqdm(total=len(list(train_df_weekly.Province_State.unique()))) as pbar:
    for province in train_df_weekly.Province_State.unique():
        province_df = train_df_weekly.query(f"Province_State=='{province}'")
        n = 0
        for i in range(0,len(province_df),int(days_in_sequence/3)):
            n += 1
            if i+days_in_sequence<=len(province_df):
                #prepare all the temporal inputs
                google_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Google_Trend.values]
                restriction_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].restriction.values]
                quarantine_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                #preparing all the demographic inputs
                demo_input = []
                for col in Sector:
                    col = float(province_df[f"{col}"].iloc[0])
                    demo_input.append(col)

                expected_trend = float(province_df.iloc[i+days_in_sequence-1].Google_Trend)
                #expected weekly claim is the sum of past 7 days including today
                expected_claims = float(province_df.iloc[i+days_in_sequence-1].UI_norm)

                trend_list.append({"google_trend":google_trend,
                                   "restriction_trend":restriction_trend,
                                   "quarantine_trend":quarantine_trend,
                                   "demographic_inputs":demo_input,
                                   "expected_trends":expected_trend,
                                   "expected_claims":expected_claims})
        pbar.update(1)
trend_df_weekly = pd.DataFrame(trend_list)
#Checkpoint: length of trend_df_weekly is 450
len(trend_df_weekly)
trend_df_daily["temporal_inputs"] = [np.asarray([trends["google_trend"],trends["restriction_trend"],trends["quarantine_trend"]]) for idx,trends in trend_df_daily.iterrows()]
trend_df_daily = shuffle(trend_df_daily)
trend_df_weekly["temporal_inputs"] = [np.asarray([trends["google_trend"],trends["restriction_trend"],trends["quarantine_trend"]]) for idx,trends in trend_df_weekly.iterrows()]
trend_df_weekly = shuffle(trend_df_weekly)
#Split training and testing data only from trend_df_weekly to avoid data leakage
sequence_length = 20
X = trend_df_weekly[["temporal_inputs","demographic_inputs"]]
y = trend_df_weekly[["expected_trends","expected_claims"]]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=617)
x_train = x_train.append(trend_df_daily[["temporal_inputs","demographic_inputs"]], sort=False)
y_train = y_train.append(trend_df_daily[["expected_trends","expected_claims"]], sort=False)
#Checkpoint length of training_item_count is 579
training_item_count = len(x_train)
validation_item_count = len(x_test)
training_item_count
#transpose the shape of x_temporal_train to (training_item_count, 20 = sequence_length, 3)
X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in x_train["temporal_inputs"].values]),(training_item_count,3,sequence_length)),(0,2,1) )).astype(np.float32)
X_demographic_train = np.asarray([np.asarray(x) for x in x_train["demographic_inputs"]]).astype(np.float32)
Y_trends_train = np.asarray([np.asarray(x) for x in y_train["expected_trends"]]).astype(np.float32)
Y_claims_train = np.asarray([np.asarray(x) for x in y_train["expected_claims"]]).astype(np.float32)
X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in x_test["temporal_inputs"]]),(validation_item_count,3,sequence_length)),(0,2,1)) ).astype(np.float32)
X_demographic_test = np.asarray([np.asarray(x) for x in x_test["demographic_inputs"]]).astype(np.float32)
Y_trends_test = np.asarray([np.asarray(x) for x in y_test["expected_trends"]]).astype(np.float32)
Y_claims_test = np.asarray([np.asarray(x) for x in y_test["expected_claims"]]).astype(np.float32)
#Kept the architecture from [Francois Lemarchand](https://www.kaggle.com/frlemarchand/covid-19-forecasting-with-an-rnn/comments#793512)'s Notebook to generate 2 seperate Losses, 
#and merged the parallel cells from Google Trend forecast to Claim output forecast

#temporal input branch
temporal_input_layer = Input(shape=(sequence_length,3))
main_rnn_layer = layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2)(temporal_input_layer)

#demographic input branch
demographic_input_layer = Input(shape=(21))
demographic_dense = layers.Dense(16)(demographic_input_layer)
demographic_dropout = layers.Dropout(0.2)(demographic_dense)

#trends output branch
rnn_t = layers.LSTM(32)(main_rnn_layer)
merge_t = layers.Concatenate(axis=-1)([rnn_t,demographic_dropout])
dense_t = layers.Dense(128)(merge_t)
dropout_t = layers.Dropout(0.3)(dense_t)
trends = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1),name="trends")(dropout_t)

#claim output branch
#change the structure here to merge dropout_t as well
rnn_c = layers.LSTM(32)(main_rnn_layer)
merge_c = layers.Concatenate(axis=-1)([rnn_c,dropout_t,demographic_dropout])
dense_c = layers.Dense(128)(merge_c)
dropout_c = layers.Dropout(0.3)(dense_c)
claims = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1), name="claims")(dropout_c)


model = Model([temporal_input_layer,demographic_input_layer], [trends,claims])

model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")
history = model.fit([X_temporal_train,X_demographic_train], [Y_trends_train, Y_claims_train], 
          epochs = 250, 
          batch_size = 16, 
          validation_data=([X_temporal_test,X_demographic_test],  [Y_trends_test, Y_claims_test]), 
          callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
plt.plot(history.history['trends_loss'])
plt.plot(history.history['val_trends_loss'])
plt.title('Loss over epochs for the number of Google Trend Search for "File for Unemployment"')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
plt.plot(history.history['claims_loss'])
plt.plot(history.history['val_claims_loss'])
plt.title('Loss over epochs for the number of initial claims')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
#Load the best model selected
model.load_weights("/kaggle/input/select-model/best_model_v32.h5")
score_train = model.evaluate([X_temporal_train,X_demographic_train], [Y_trends_train, Y_claims_train], verbose=0)
score_test = model.evaluate([X_temporal_test,X_demographic_test], [Y_trends_test, Y_claims_test], verbose=0)
print(f"train MSLE score is {score_train}".format(score_train))
print(f"test MSLE score is {score_test}".format(score_test))
predictions = model.predict([X_temporal_test,X_demographic_test])
display_limit = 30
for inputs, pred_trends, exp_trends, pred_claims, exp_claims in zip(X_temporal_test,predictions[0][:display_limit], Y_trends_test[:display_limit], predictions[1][:display_limit], Y_claims_test[:display_limit]):
    print("================================================")
    print(inputs)
    print("Expected trends:", exp_trends, " Prediction:", pred_trends[0], "Expected Claims:", exp_claims, " Prediction:", pred_claims[0] )
#Prepare input data for forecasting
def build_inputs_for_date(province, date, df):
    
    start_date = date - timedelta(days=20)
    end_date = date - timedelta(days=1)
    
    str_start_date = start_date.strftime("%Y-%m-%d")
    str_end_date = end_date.strftime("%Y-%m-%d")

    df = df.query("Province_State=='"+province+"' and Date>='"+str_start_date+"' and Date<='"+str_end_date+"'")
    #print(df)
    
    #preparing the temporal inputs
    temporal_input_data = np.transpose(np.reshape(np.asarray([df["Google_Trend"],
                                                 df["restriction"],
                                                 df["quarantine"]]),
                                     (3,sequence_length)), (1,0) ).astype(np.float32)
    
    demographic_input_data = []
    for col in Sector:
        col = float(df[f"{col}"].iloc[0])
        demographic_input_data.append(col)
    
    return [np.array([temporal_input_data]), np.array([demographic_input_data])]
#Generate forecast and append predicted google trend into input data
def predict_for_region(province, df):
    # start from 2020-04-19
    restriction_end_date = datetime.strptime("2020-07-04","%Y-%m-%d")
    quarantine_end_date = datetime.strptime("2020-06-15","%Y-%m-%d")
    
    begin_prediction = "2020-04-19"
    start_date = datetime.strptime(begin_prediction,"%Y-%m-%d")
    end_prediction = "2020-06-30"
    end_date = datetime.strptime(end_prediction,"%Y-%m-%d")
    
    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days+1)]
    
    df["Date"] = pd.to_datetime(df["Date"])
    for date in date_list:
        
        input_data = build_inputs_for_date(province, date, df)
        result = model.predict(input_data)
        
        COVID_df = quarantine_data_df.query(f"Province_State=='{province}'")[["SAH_end","Restriction_Easing"]]
        
        if date <= datetime.strptime("2020-05-09","%Y-%m-%d"):
            G_trend = G_trend_all.query(f"Province_State=='{province}' and Date == '{date}'").iloc[0][2]
        else:
            G_trend = result[0][0][0]
        
        #print(G_trend)
        
        quarantine = 0 if COVID_df.iloc[0][0]<date else 1
        restriction = 0 if COVID_df.iloc[0][1]<date else 1
        ui_base = UI_base.query("Province_State=='"+province+"'").iloc[0][1]
        

        df = df.append({"Province_State":province, 
                        "Date":date.strftime("%Y-%m-%d"), 
                        #"restriction": 0 if date >restriction_end_date else input_data[0][0][-1][1],
                        #"quarantine": 0 if date >quarantine_end_date else input_data[0][0][-1][2],
                        "restriction": 0 if date >restriction_end_date else restriction,
                        "quarantine": 0 if date >quarantine_end_date else quarantine,
                        "UI_base": ui_base,
                        #"Google_Trend": 0 if result[0][0][0]<0 else result[0][0][0],
                        "Google_Trend": 0 if G_trend <0 else G_trend,
                        "UI_norm": result[1][0][0],
                        #"Pop_above_18":input_data[1][0][0],
                        "Agriculture":input_data[1][0][0],
                        "Energy":input_data[1][0][1],
                        "Utilities":input_data[1][0][2],
                        "Construction":input_data[1][0][3],
                        "Durable":input_data[1][0][4],
                        "Nondurable":input_data[1][0][5],
                        "Wholesale":input_data[1][0][6],
                        "Retail":input_data[1][0][7],
                        "Transportation":input_data[1][0][8],
                        "Information":input_data[1][0][9],
                        "Finance":input_data[1][0][10],
                        "Real-estate":input_data[1][0][11],
                        "Technology":input_data[1][0][12],
                        "Management":input_data[1][0][13],
                        "Services":input_data[1][0][14],
                        "Education":input_data[1][0][15],
                        "Health-care":input_data[1][0][16],
                        "Entertainment":input_data[1][0][17],
                        "Food":input_data[1][0][18],
                        "Other":input_data[1][0][19],
                        "Gov":input_data[1][0][20]},
                       ignore_index=True)
        
        df["Date"] = pd.to_datetime(df["Date"])
            
    return df
#Forecast using latest Google Trend data
New_Google_Trend = gpd.read_file("/kaggle/input/forecast-0509/Google_trend_update_0509.csv").drop(["geometry"],axis=1)
New_Google_Trend["Date"] = pd.to_datetime(New_Google_Trend["Date"])
New_Google_Trend = New_Google_Trend.query("Date >= '2020-05-04'")
New_Google_Trend["Google_Trend"] = New_Google_Trend["Google_Trend"].astype("float")
G_trend_all = Google_trend_df.append(New_Google_Trend)
forecast_weekly = train_df.merge(train_all,
                          on=["Province_State", "Date"],
                          how="left",
                          )
forecast_weekly['Initial_Claims'] = forecast_weekly['Initial_Claims'].astype("float")
forecast_weekly['UI_norm'] = forecast_weekly['Initial_Claims']/(forecast_weekly['UI_base']/13)
forecast_weekly = forecast_weekly.query("Date>='2020-01-26' and Date<='2020-04-18'")
#Generate forecast
forecast_df = forecast_weekly
with tqdm(total=len(list(forecast_df.Province_State.unique()))) as pbar:
    for province in forecast_df.Province_State.unique():
        forecast_df = predict_for_region(province, forecast_df)
        pbar.update(1)
forecast_df.to_csv("forecast_result_0509.csv")
begin_actual = "2020-02-01"
start_date = datetime.strptime(begin_actual,"%Y-%m-%d")
end_prediction = "2020-06-30"
end_date = datetime.strptime(end_prediction,"%Y-%m-%d")
date_list = [start_date + timedelta(days=7*x) for x in range(int((end_date-start_date).days/7)+1)]
forecast_df['UI'] = forecast_df['UI_norm']*forecast_df['UI_base']/13
forecast_df_v2 = forecast_df[forecast_df["Date"].isin(date_list)][["Date","Province_State","UI"]]
#Create summarized actuals for National Wide result
Actuals = forecast_df.groupby(['Date'])['UI'].sum().reset_index()
Actuals = Actuals[Actuals["Date"].isin(date_list)]
Actuals["Province_State"] ="US"
forecast_df_v2 = forecast_df_v2.append(Actuals,sort=False)
def plot_prediction(province, forcast_start):
    copy_df_new = forecast_df_v2.query("Province_State=='"+province+"'")
    copy_df_new["Actual"] = [row.UI if row.Date <= forcast_start else None for idx,row in copy_df_new.iterrows()]
    copy_df_new["Prediction"] = [row.UI if row.Date >= forcast_start else None for idx,row in copy_df_new.iterrows()]
    copy_df_new.iloc[int((datetime.strptime("2020-04-25","%Y-%m-%d")-start_date).days/7),3] = UI_0425[province]
    copy_df_new.iloc[int((datetime.strptime("2020-05-02","%Y-%m-%d")-start_date).days/7),3] = UI_0502[province]
    copy_df_new["Date"] = copy_df_new["Date"].dt.strftime("%m/%d/%Y")
    print(copy_df_new)
    plt.plot(copy_df_new.Actual.values)
    plt.plot(copy_df_new.Prediction.values)
    plt.title(f"{province} Unemployment Initial Claims Forecast".format(province))
    plt.ylabel('Number of initial claims')
    plt.xlabel('Date')
    plt.xticks(range(len(copy_df_new.Date.values)),copy_df_new.Date.values,rotation='vertical')
    plt.legend(['Actual', 'Prediction'], loc='best')
    plt.show()
####Add latest actual data starting from 04-25
UI_0425 = {
    "US": 3846000,
    "New York" : 222040,
    "California" : 325343,
    "Pennsylvania" : 114700,
    "Michigan" : 82004,
    "Illinois" : 81596,
    "Florida": 433103,
    "Texas": 254084,
    "Georgia": 266565
    }
UI_0502 = {
    "US": 3169000,
    "New York" : 197607,
    "California" : None,
    "Pennsylvania" : 81779,
    "Michigan" : None,
    "Illinois" : 74476,
    "Florida": None,
    "Texas": None,
    "Georgia": None
    }
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
for key in UI_0502:
    plt.figure()
    plot_prediction(key, datetime.strptime("2020-04-18","%Y-%m-%d"))
