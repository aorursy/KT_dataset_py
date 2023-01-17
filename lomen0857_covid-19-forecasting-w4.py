import numpy as np

import pandas as pd

import geopandas as gpd

from shapely.geometry import Point

import os

import tensorflow as tf

from tqdm import tqdm

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_log_error



from datetime import datetime

from datetime import timedelta



from tensorflow.keras.optimizers import Adagrad

from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

train_df["ConfirmedCases"] = train_df["ConfirmedCases"].astype("float")

train_df["Fatalities"] = train_df["Fatalities"].astype("float")

train_df["Country_Region"] = [ row.Country_Region.replace("'","").strip(" ") if row.Province_State=="" else str(row.Country_Region+"_"+row.Province_State).replace("'","").strip(" ") for idx,row in train_df.iterrows()]
extra_data_df = gpd.read_file("/kaggle/input/covid19-enriched-dataset-week-2/enriched_covid_19_week_2.csv")

                        

extra_data_df["Country_Region"] = [country_name.replace("'","") for country_name in extra_data_df["Country_Region"]]

extra_data_df["restrictions"] = extra_data_df["restrictions"].astype("int")

extra_data_df["quarantine"] = extra_data_df["quarantine"].astype("int")

extra_data_df["schools"] = extra_data_df["schools"].astype("int")

extra_data_df["total_pop"] = extra_data_df["total_pop"].astype("float")

extra_data_df["density"] = extra_data_df["density"].astype("float")

extra_data_df["hospibed"] = extra_data_df["hospibed"].astype("float")

extra_data_df["lung"] = extra_data_df["lung"].astype("float")

extra_data_df["total_pop"] = extra_data_df["total_pop"]/max(extra_data_df["total_pop"])

extra_data_df["density"] = extra_data_df["density"]/max(extra_data_df["density"])

extra_data_df["hospibed"] = extra_data_df["hospibed"]/max(extra_data_df["hospibed"])

extra_data_df["lung"] = extra_data_df["lung"]/max(extra_data_df["lung"])

extra_data_df["age_100+"] = extra_data_df["age_100+"].astype("float")

extra_data_df["age_100+"] = extra_data_df["age_100+"]/max(extra_data_df["age_100+"])



extra_data_df = extra_data_df[["Country_Region","Date","restrictions","quarantine","schools","hospibed","lung","total_pop","density","age_100+"]]
ll_df = pd.read_csv("/kaggle/input/country-region-lat-long/Country_Region_Lat_Long.csv")

ll_df = ll_df[["Country_Region","Lat","Long","Lat(//15)","Long(//30)"]]

ll_df = ll_df.drop_duplicates().reset_index(drop=True)

ll_df["Lat"] = ll_df["Lat"]+90

ll_df["Lat"] = ll_df["Lat"]//15

ll_df["Lat"] = ll_df["Lat"]/max(ll_df["Lat"])



ll_df["Long"] = ll_df["Long"]+180

ll_df["Long"] = ll_df["Long"]//30

ll_df["Long"] = ll_df["Long"]/max(ll_df["Long"])



#ll_df["Lat"] = ll_df["Lat(//15)"]/max(ll_df["Lat(//15)"])

#ll_df["Long"] = ll_df["Long(//30)"]/max(ll_df["Long(//30)"])
ll_df
extra_data_df = extra_data_df.merge(ll_df, how="left", on=["Country_Region"]).drop_duplicates()
train_df = train_df.merge(extra_data_df, how="left", on=['Country_Region','Date']).drop_duplicates()
for country_region in train_df.Country_Region.unique():

    query_df = train_df.query("Country_Region=='"+country_region+"' and Date=='2020-03-25'")

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"total_pop"] = query_df.total_pop.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"hospibed"] = query_df.hospibed.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"density"] = query_df.density.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"lung"] = query_df.lung.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"Lat"] = query_df.Lat.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"Long"] = query_df.Long.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"age_100+"] = query_df["age_100+"].values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"restrictions"] = query_df.restrictions.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"quarantine"] = query_df.quarantine.values[0]

    train_df.loc[(train_df["Country_Region"]==country_region) & (train_df["Date"]>"2020-03-25"),"schools"] = query_df.schools.values[0]
train_df
median_pop = np.median(extra_data_df.total_pop)

median_hospibed = np.median(extra_data_df.hospibed)

median_density = np.median(extra_data_df.density)

median_lung = np.median(extra_data_df.lung)

median_lat = np.median(extra_data_df.Lat)

median_long = np.median(extra_data_df.Long)

median_centenarian_pop = np.median(extra_data_df["age_100+"])



for country_region in train_df.Country_Region.unique():

    if extra_data_df.query("Country_Region=='"+country_region+"'").empty:

        

        train_df.loc[train_df["Country_Region"]==country_region,"total_pop"] = median_pop

        train_df.loc[train_df["Country_Region"]==country_region,"hospibed"] = median_hospibed

        train_df.loc[train_df["Country_Region"]==country_region,"density"] = median_density

        train_df.loc[train_df["Country_Region"]==country_region,"lung"] = median_lung

        train_df.loc[train_df["Country_Region"]==country_region,"Lat"] = median_lat

        train_df.loc[train_df["Country_Region"]==country_region,"Long"] = median_long

        train_df.loc[train_df["Country_Region"]==country_region,"age_100+"] = median_centenarian_pop

        train_df.loc[train_df["Country_Region"]==country_region,"restrictions"] = 0

        train_df.loc[train_df["Country_Region"]==country_region,"quarantine"] = 0

        train_df.loc[train_df["Country_Region"]==country_region,"schools"] = 0
trend_df = pd.DataFrame(columns={"infection_trend","fatality_trend","quarantine_trend","school_trend","total_population","expected_cases","expected_fatalities"})
train_df = train_df.query("Date>'2020-01-21'and Date<'2020-04-15'")

days_in_sequence = 21



trend_list = []



with tqdm(total=len(list(train_df.Country_Region.unique()))) as pbar:

    for country in train_df.Country_Region.unique():

        for province in train_df.query(f"Country_Region=='{country}'").Province_State.unique():

            province_df = train_df.query(f"Country_Region=='{country}' and Province_State=='{province}'")

            

            #I added a quick hack to double the number of sequences

            #Warning: This will later create a minor leakage from the 

            # training set into the validation set.

            for i in range(0,len(province_df),int(days_in_sequence/3)):

                if i+days_in_sequence<=len(province_df):

                    #prepare all the temporal inputs

                    infection_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].ConfirmedCases.values]

                    fatality_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].Fatalities.values]

                    restriction_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].restrictions.values]

                    quarantine_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].quarantine.values]

                    school_trend = [float(x) for x in province_df[i:i+days_in_sequence-1].schools.values]



                    #preparing all the demographic inputs

                    total_population = float(province_df.iloc[i].total_pop)

                    density = float(province_df.iloc[i].density)

                    hospibed = float(province_df.iloc[i].hospibed)

                    lung = float(province_df.iloc[i].lung)

                    lat = float(province_df.iloc[i].Lat)

                    long = float(province_df.iloc[i].Long)

                    centenarian_pop = float(province_df.iloc[i]["age_100+"])



                    expected_cases = float(province_df.iloc[i+days_in_sequence-1].ConfirmedCases)

                    expected_fatalities = float(province_df.iloc[i+days_in_sequence-1].Fatalities)



                    trend_list.append({"infection_trend":infection_trend,

                                     "fatality_trend":fatality_trend,

                                     "restriction_trend":restriction_trend,

                                     "quarantine_trend":quarantine_trend,

                                     "school_trend":school_trend,

                                     "demographic_inputs":[total_population,density,hospibed,lung,lat,long,centenarian_pop],

                                     "expected_cases":expected_cases,

                                     "expected_fatalities":expected_fatalities})

        pbar.update(1)

trend_df = pd.DataFrame(trend_list)
trend_df["temporal_inputs"] = [np.asarray([trends["infection_trend"],trends["fatality_trend"],trends["restriction_trend"],trends["quarantine_trend"],trends["school_trend"]]) for idx,trends in trend_df.iterrows()]



trend_df = shuffle(trend_df)
i=0

temp_df = pd.DataFrame()

for idx,row in trend_df.iterrows():

    if sum(row.infection_trend)>0:

        temp_df = temp_df.append(row)

    else:

        if i<25:

            temp_df = temp_df.append(row)

            i+=1

trend_df = temp_df
sequence_length = 20

training_percentage = 0.9
training_item_count = int(len(trend_df)*training_percentage)

validation_item_count = len(trend_df)-int(len(trend_df)*training_percentage)

training_df = trend_df[:training_item_count]

validation_df = trend_df[training_item_count:]
X_temporal_train = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in training_df["temporal_inputs"].values]),(training_item_count,5,sequence_length)),(0,2,1) )).astype(np.float32)

X_demographic_train = np.asarray([np.asarray(x) for x in training_df["demographic_inputs"]]).astype(np.float32)

Y_cases_train = np.asarray([np.asarray(x) for x in training_df["expected_cases"]]).astype(np.float32)

Y_fatalities_train = np.asarray([np.asarray(x) for x in training_df["expected_fatalities"]]).astype(np.float32)
X_temporal_test = np.asarray(np.transpose(np.reshape(np.asarray([np.asarray(x) for x in validation_df["temporal_inputs"]]),(validation_item_count,5,sequence_length)),(0,2,1)) ).astype(np.float32)

X_demographic_test = np.asarray([np.asarray(x) for x in validation_df["demographic_inputs"]]).astype(np.float32)

Y_cases_test = np.asarray([np.asarray(x) for x in validation_df["expected_cases"]]).astype(np.float32)

Y_fatalities_test = np.asarray([np.asarray(x) for x in validation_df["expected_fatalities"]]).astype(np.float32)
#temporal input branch

temporal_input_layer = Input(shape=(sequence_length,5))

main_rnn_layer = layers.LSTM(100, return_sequences=True, recurrent_dropout=0.2)(temporal_input_layer)



#demographic input branch

demographic_input_layer = Input(shape=(7))

demographic_dense = layers.Dense(20)(demographic_input_layer)

demographic_dropout = layers.Dropout(0.2)(demographic_dense)



#cases output branch

rnn_c = layers.LSTM(50)(main_rnn_layer)

merge_c = layers.Concatenate(axis=-1)([rnn_c,demographic_dropout])

dense_c = layers.Dense(150)(merge_c)

dropout_c = layers.Dropout(0.2)(dense_c)

cases = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1),name="cases")(dropout_c)



#fatality output branch

rnn_f = layers.LSTM(50)(main_rnn_layer)

merge_f = layers.Concatenate(axis=-1)([rnn_f,demographic_dropout])

dense_f = layers.Dense(150)(merge_f)

dropout_f = layers.Dropout(0.2)(dense_f)

fatalities = layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1), name="fatalities")(dropout_f)





model = Model([temporal_input_layer,demographic_input_layer], [cases,fatalities])



model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),

             EarlyStopping(monitor='val_loss', patience=30),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer=opt)
history = model.fit([X_temporal_train,X_demographic_train], [Y_cases_train, Y_fatalities_train], 

          epochs = 300, 

          batch_size = 10, 

          validation_data=([X_temporal_test,X_demographic_test],  [Y_cases_test, Y_fatalities_test]), 

          callbacks=callbacks)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
predictions = model.predict([X_temporal_test,X_demographic_test])
len(predictions[0][0])
train_df = train_df.query("Date>'2020-01-21'and Date<'2020-04-01'")
#Will retrieve the number of cases and fatalities for the past 6 days from the given date

def build_inputs_for_date(country, province, date, df):

    start_date = date - timedelta(days=20)

    end_date = date - timedelta(days=1)

    

    str_start_date = start_date.strftime("%Y-%m-%d")

    str_end_date = end_date.strftime("%Y-%m-%d")

    df = df.query("Country_Region=='"+country+"' and Province_State=='"+province+"' and Date>='"+str_start_date+"' and Date<='"+str_end_date+"'")

    #preparing the temporal inputs

    temporal_input_data = np.transpose(np.reshape(np.asarray([df["ConfirmedCases"],

                                                 df["Fatalities"],

                                                 df["restrictions"],

                                                 df["quarantine"],

                                                 df["schools"]]),

                                     (5,sequence_length)), (1,0) ).astype(np.float32)

    

    #preparing all the demographic inputs

    total_population = float(province_df.iloc[i].total_pop)

    density = float(province_df.iloc[i].density)

    hospibed = float(province_df.iloc[i].hospibed)

    lung = float(province_df.iloc[i].lung)

    lat = float(province_df.iloc[i].Lat)

    long = float(province_df.iloc[i].Long)

    

    centenarian_pop = float(province_df.iloc[i]["age_100+"])

    demographic_input_data = [total_population,density,hospibed,lung,lat,long,centenarian_pop]

    

    return [np.array([temporal_input_data]), np.array([demographic_input_data])]
#Take a dataframe in input, will do the predictions and return the dataframe with extra rows

#containing the predictions

def predict_for_region(country, province, df):

    begin_prediction = "2020-04-01"

    start_date = datetime.strptime(begin_prediction,"%Y-%m-%d")

    end_prediction = "2020-05-14"

    end_date = datetime.strptime(end_prediction,"%Y-%m-%d")

    

    date_list = [start_date + timedelta(days=x) for x in range((end_date-start_date).days+1)]

    for date in date_list:

        input_data = build_inputs_for_date(country, province, date, df)

        result = model.predict(input_data)

        

        #just ensuring that the outputs is

        #higher than the previous counts

        result[0] = np.round(result[0])

        if result[0]<input_data[0][0][-1][0]:

            result[0]=np.array([[input_data[0][0][-1][0]]])

        

        result[1] = np.round(result[1])

        if result[1]<input_data[0][0][-1][1]:

            result[1]=np.array([[input_data[0][0][-1][1]]])

        

        #We assign the quarantine and school status

        #depending on previous values

        #e.g Once a country is locked, it will stay locked until the end

        df = df.append({"Country_Region":country, 

                        "Province_State":province, 

                        "Date":date.strftime("%Y-%m-%d"), 

                        "restrictions": 1 if any(input_data[0][0][2]) else 0,

                        "quarantine": 1 if any(input_data[0][0][3]) else 0,

                        "schools": 1 if any(input_data[0][0][4]) else 0,

                        "total_pop": input_data[1][0],

                        "density": input_data[1][0][1],

                        "hospibed": input_data[1][0][2],

                        "lung": input_data[1][0][3],

                        "lat": input_data[1][0][4],

                        "long": input_data[1][0][5],

                        

                        "age_100+": input_data[1][0][6],

                        "ConfirmedCases":round(result[0][0][0]),	

                        "Fatalities":round(result[1][0][0])},

                       ignore_index=True)

    return df
#The functions that are called here need to optimise, sorry about that!

copy_df = train_df

with tqdm(total=len(list(copy_df.Country_Region.unique()))) as pbar:

    for country in copy_df.Country_Region.unique():

        for province in copy_df.query("Country_Region=='"+country+"'").Province_State.unique():

            copy_df = predict_for_region(country, province, copy_df)

        pbar.update(1)
groundtruth_df = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/train.csv")



groundtruth_df["ConfirmedCases"] = groundtruth_df["ConfirmedCases"].astype("float")

groundtruth_df["Fatalities"] = groundtruth_df["Fatalities"].astype("float")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to this Dataframe.

groundtruth_df["Country_Region"] = [ row.Country_Region.replace("'","").strip(" ") if row.Province_State=="" else str(row.Country_Region+"_"+row.Province_State).replace("'","").strip(" ") for idx,row in groundtruth_df.iterrows()]



last_date = groundtruth_df.Date.unique()[-1]
#to remove annoying warnings from pandas

pd.options.mode.chained_assignment = None



def get_RMSLE_per_region(region, groundtruth_df, display_only=False):

    groundtruth_df["ConfirmedCases"] = groundtruth_df["ConfirmedCases"].astype("float")

    groundtruth_df["Fatalities"] = groundtruth_df["Fatalities"].astype("float")

    

    #we only take data until the 30th of March 2020 as the groundtruth was not available for later dates.

    groundtruth = groundtruth_df.query("Country_Region=='"+region+"' and Date>='2020-04-01' and Date<='"+last_date+"'")

    predictions = copy_df.query("Country_Region=='"+region+"' and Date>='2020-04-01' and Date<='"+last_date+"'")

    

    RMSLE_cases = np.sqrt(mean_squared_log_error( groundtruth.ConfirmedCases.values, predictions.ConfirmedCases.values ))

    RMSLE_fatalities = np.sqrt(mean_squared_log_error( groundtruth.Fatalities.values, predictions.Fatalities.values ))

    

    if display_only:

        print(region)

        print("RMSLE on cases:",np.mean(RMSLE_cases))

        print("RMSLE on fatalities:",np.mean(RMSLE_fatalities))

    else:

        return RMSLE_cases, RMSLE_fatalities
def get_RMSLE_for_all_regions(groundtruth_df):

    RMSLE_cases_list = []

    RMSLE_fatalities_list = []

    for region in groundtruth_df.Country_Region.unique():

        RMSLE_cases, RMSLE_fatalities = get_RMSLE_per_region(region, groundtruth_df, False)

        RMSLE_cases_list.append(RMSLE_cases)

        RMSLE_fatalities_list.append(RMSLE_fatalities)

    print("RMSLE on cases:",np.mean(RMSLE_cases_list))

    print("RMSLE on fatalities:",np.mean(RMSLE_fatalities_list))
get_RMSLE_for_all_regions(groundtruth_df)
test_df = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

#The country_region got modifying in the enriched dataset by @optimo, 

# so we have to apply the same change to the test Dataframe.

test_df["Country_Region"] = [ row.Country_Region if row.Province_State=="" else row.Country_Region+"_"+row.Province_State for idx,row in test_df.iterrows() ]

test_df
submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])

with tqdm(total=len(test_df)) as pbar:

    for idx, row in test_df.iterrows():

        #Had to remove single quotes because of countries like Cote D'Ivoire for example

        country_region = row.Country_Region.replace("'","").strip(" ")

        province_state = row.Province_State.replace("'","").strip(" ")

        item = copy_df.query("Country_Region=='"+country_region+"' and Province_State=='"+province_state+"' and Date=='"+row.Date+"'")

        submission_df = submission_df.append({"ForecastId":row.ForecastId,

                                              "ConfirmedCases":int(item.ConfirmedCases.values[0]),

                                              "Fatalities":int(item.Fatalities.values[0])},

                                             ignore_index=True)

        pbar.update(1)
submission_df.to_csv("submission.csv",index=False)